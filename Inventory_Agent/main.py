import asyncio
from dataclasses import dataclass
from typing import Optional
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, function_tool, enable_verbose_stdout_logging
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

enable_verbose_stdout_logging()

load_dotenv()
set_tracing_disabled(disabled=True)


gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(disabled=True)

gemini_client = AsyncOpenAI (
     api_key = gemini_api_key,
     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
     model="gemini-2.0-flash",
     openai_client=gemini_client
 )

# Sample stock list
stock_list = [
    {"id": 1, "item": "Laptop", "qty": 10},
    {"id": 2, "item": "Mouse", "qty": 50},
    {"id": 3, "item": "Keyboard", "qty": 30}
]

@dataclass
class StockInput:
    action: str
    id: int = None
    item: str = None
    qty: int = None

class AgentReply(BaseModel):
    reply_type: str  # e.g. "info" or "inventory"
    details: str = None

# Tool for managing stock
@function_tool
async def handleStock(data) -> str:
    """
    This function helps in stock control.
    Supported actions: 'add' (insert new), 'update' (modify existing), 'delete' (remove entry).
    For 'add' you must provide item and qty. 
    For 'update' or 'delete' you must provide the id.
    """
    global stock_list
    action = data.action.lower()

    print("\n\n\n")
    print(f"Action: {action}, Data: {data}")
    if action == "add":
        if not data.item or data.qty is None:
            return "Error: Item name and quantity are required to add."
        new_id = max([x["id"] for x in stock_list], default=0) + 1
        stock_list.append({"id": new_id, "item": data.item, "qty": data.qty})
        return f"Added {data.item} with ID {new_id} and quantity {data.qty}."

    elif action == "update":
        if data.id is None or not data.item or data.qty is None:
            return "Error: ID, item and qty are required to update."
        for s in stock_list:
            if s["id"] == data.id:
                s["item"] = data.item
                s["qty"] = data.qty
                return f"Updated item ID {data.id} to {data.item} with quantity {data.qty}."
        return f"Error: Item with ID {data.id} not found."

    elif action == "delete":
        if data.id is None:
            return "Error: ID is required to delete."
        for i, s in enumerate(stock_list):
            if s["id"] == data.id:
                removed = stock_list.pop(i)
                return f"Deleted item ID {data.id} ({removed['item']})."
        return f"Error: Item with ID {data.id} not found."

    else:
        return "Error: Invalid action. Use 'add', 'update', or 'delete'."

# Define the agent
agent = Agent(
    name="Inventory Helper",
    instructions="You are an inventory assistant. For stock tasks (add, update, delete), ALWAYS use handleStock tool and return the updated stock list.",
    model=model,
    tools=[handleStock],
    output_type=AgentReply
)

async def main(startMessage: str):
    print(f"RUN Started: {startMessage}")
    
    result = await Runner.run(
        agent,
        input=startMessage
    )
    print(result.final_output)
    
    if result.final_output and result.final_output.reply_type == "inventory":
        print("\nCurrent Stock:")
        print(stock_list)

def start():
    asyncio.run(main("Add a new stock item: Monitor, quantity 20"))

if __name__ == "__main__":
  start()