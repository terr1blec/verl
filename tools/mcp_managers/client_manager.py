# This class use the Qwen MCPManager as a reference.
import asyncio
import json
import threading
import os
import jsonlines
from fastmcp import Client
from fastmcp.exceptions import ToolError


class MCPClientManager:
    _instance = None  # Private class variable to store the unique instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPClientManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'clients'):  # The singleton should only be inited once
            self.clients = {}
            self.class_to_path_mapping = {}
            self.log_info = {}
            
            # Set up a new event loop in a separate thread for async operations
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.start_loop, daemon=True)
            self.loop_thread.start()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # TODO
    def is_valid_mcp_servers(self, config: dict): 
        return True

    def initConfig(self, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        
        if not self.is_valid_mcp_servers(config):
            raise ValueError('Config of mcpservers is not valid')
        
        future = asyncio.run_coroutine_threadsafe(self.init_config_async(config), self.loop)
        try:
            result = future.result()
            return result
        except Exception as e:
            raise e

    async def init_config_async(self, config: dict):
        self.tool_schemas = []
        self.tools = {}

        def get_tool_schema(tool_class, tool):
            return {
                "type": "function",
                "function": {
                    "name": f"{tool_class}-{tool.name}",
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }

        mcp_servers = config['mcpServers']
        for server_name, server in mcp_servers.items(): # server name is class name
            self.class_to_path_mapping[server_name] = server['local_path']
            tmp_client = Client(server['local_path'])
            async with tmp_client:
                tmp_tools = await tmp_client.list_tools()

                for tool in tmp_tools:
                    tool_schema = get_tool_schema(server_name, tool)
                    tool_name = f"{server_name}-{tool.name}"
                    self.tool_schemas.append(tool_schema)
                    self.tools.update({tool_name: tool_schema})

            await tmp_client.close() # close temporary client

    def dump_log(self, log_dump_path="log/log.jsonl") -> None:
        try:
            os.makedirs(os.path.dirname(log_dump_path), exist_ok=True)
            
            with jsonlines.open(log_dump_path, mode='a') as writer:
                for client_id, logs in self.log_info.items():
                    writer.write({client_id: logs})
            
        except Exception as e:
            print(f"Error dumping logs: {e}")

    def add_log(self, client_id, info: dict) -> None:
        """
        info
            - chat
                - system
                - user
                - assistant
            - tool
                - tool_name
                - tool_args
                - tool_result
        """
        no_class_client_id = client_id.split("-")[-1] # remove tool class from client id
        if no_class_client_id not in self.log_info:
            self.log_info[no_class_client_id] = []
        
        log = {}
        if "chat" in info:
            log["chat"] = {
                # "system": info['chat'].get("system", ""),
                "user": info['chat'].get("user", ""),
                "assistant": info['chat'].get("assistant", "")
            }
        
        if "tool" in info:
            log["tool"] = {
                "tool_name": info["tool"].get("tool_name", ""),
                "tool_args": info["tool"].get("tool_args", ""),
                "tool_result": info["tool"].get("tool_result", "")
            }
        
        if log:
            self.log_info[no_class_client_id].append(log)
            
    def set_status(self, client_id):
        client_info = self.clients.get(client_id)
        if client_info:
            self.clients[client_id]['status'] = True

    def get_client(self, client_id) -> tuple[Client, bool]:
        """
        Get a client from a given client id or create a new client if not exist.
        """
        if self.clients.get(client_id, None):
            return self.clients[client_id]['client'], self.clients[client_id]['status']

        tool_class = client_id.split("-")[0]
        new_client = Client(self.class_to_path_mapping[tool_class])
        self.clients[client_id] = {
            "client": new_client,
            "status": False, # not initialized
        }
        return new_client, False

    async def close_client(self, client_id):
        """Close and remove a client from the manager"""
        client_info = self.clients.get(client_id)
        if client_info:
            try:
                await client_info["client"].close()
            except Exception as e:
                print(f"Error closing client {client_id}: {e}")
            finally:
                del self.clients[client_id]
                print(f"Client {client_id} closed and removed")

    def close_all_clients(self, ignore_stateless_client: bool = False):
        """Close all clients"""
        futures = []
        for client_id in list(self.clients.keys()):
            future = asyncio.run_coroutine_threadsafe(self.close_client(client_id), self.loop)
            futures.append(future)
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error closing client: {e}")
                
        self.log_info = {}


    def save_all_scenario(self) -> dict:
        saved_all_scenario = {}
        for client_id, client_info in self.clients.items():
            try:
                tool_class = client_id.split("-")[0]
                saved_scenario = self.call_tool(
                    client_id = client_id,
                    tool_name = "save_scenario",
                    tool_args = {},
                )
                saved_scenario = json.loads(saved_scenario)
            except:
                saved_scenario = None
            saved_all_scenario.update({tool_class: saved_scenario})
        return saved_all_scenario

    def load_scenario(self, client_id: str, scenario: dict | None = None, check: bool = False):
        """
        Synchronous wrapper for the async call_tool method
        Args:
            client_id:
            scenario:
            check:
        """
        client, status = self.get_client(client_id)
        if not status and scenario:
            tool_args = {"scenario": scenario}
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async("load_scenario", tool_args, client, client_id),
                self.loop,
            )
            try:
                result = future.result()
                if check: # TODO: we need to handle the case where the scenario fails to load 
                    saved_scenario = self.call_tool(
                        client_id = client_id,
                        tool_name = "save_scenario",
                        tool_args = {},
                    )
                    try:
                        if scenario == json.loads(saved_scenario):
                            self.set_status(client_id)
                            print(f"Load scenario succeeded with checking: {result}")
                        else:
                            print(f"Load scenario failed. The loaded scenario mismatch with saved scenario.")
                    except:
                        print(f"Load scenario failed. The loaded scenario mismatch with saved scenario.")
                else:
                    self.set_status(client_id)
                    print(f"Load scenario succeeded without checking: {result}")
                return result
            except Exception as e:
                print(f"Load scenario failed: {e}")
                raise e
        return "This client is already initialized. Skipping..."

    def call_tool(self, tool_name, tool_args, client_id):
        """Synchronous wrapper for the async call_tool method"""
        # Use build-in load_scenario tool call for better error handling.
        if "load_scenario" in tool_name:
            return self.load_scenario(client_id = client_id, scenario = tool_args)
        
        client, status = self.get_client(client_id)
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, tool_args, client, client_id), self.loop
        )
        try:
            result = future.result()
            print(f"{tool_name} execute: {result}")
            return result
        except ToolError as e:
            print(f"{tool_name} fail before execution: {e}")
        except Exception as e:
            print(f"{tool_name} raise an unexpected error: {e}")

    async def _call_tool_async(self, tool_name: str, tool_args: dict | str, client: Client, client_id: str) -> str:
        tool_name = tool_name.split("-", 1)[-1]
        async with client:
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            result = await client.call_tool(tool_name, tool_args)

        self.add_log(
            client_id = client_id,
            info = {
                "tool": {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": result.content[0].text,
                }
            }
        )

        return result.content[0].text

    def shutdown(self, timeout=5):
        """Cleanup and shutdown the manager"""
        # Close all clients
        self.close_all_clients()
        
        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=timeout)
        
        # Handle case where thread doesn't join within timeout
        if self.loop_thread.is_alive():
            print(f"Warning: Event loop thread did not terminate within {timeout} seconds")
