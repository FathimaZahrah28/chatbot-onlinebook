import google.generativeai as genai
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages.tool import ToolMessage
from collections.abc import Iterable
from random import randint

my_api_key = "AIzaSyCmS4s2HfasXfVHwLpG0PFnZ0B5NESOxB0"
os.environ["GOOGLE_API_KEY"] = my_api_key
genai.configure(api_key=my_api_key)

class OrderState(TypedDict):
    """State representing the customer's order conversation."""
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

# System instruction
BOOKSTOREBOT_SYSINT = (
    "system",
    "You are a BookStoreBot, an interactive bookstore ordering system. A human will talk to you about the "
    "available books you have and you will answer any questions about the book catalog (and only about "
    "the catalog - no off-topic discussion, but you can chat about the books, their authors, and their background). "
    "The customer will place an order for 1 or more books from the catalog, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user). "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with book titles and categories from the CATALOG before adding them to the order. "
    "If you are unsure a book title or category matches those on the CATALOG, ask a question to clarify or redirect. "
    "You only have the books listed in the catalog. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
    "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!"
)

WELCOME_MSG = "Selamat datang di BookStoreBot! Bagaimana saya bisa membantu Anda menemukan buku hari ini?"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

@tool
def get_catalog() -> str:
    """Provide the latest up-to-date book catalog."""
    return """
    CATALOG:
    
    Fiction:
    - The Great Gatsby (F. Scott Fitzgerald)
    - To Kill a Mockingbird (Harper Lee)
    - 1984 (George Orwell)
    - The Midnight Library (Matt Haig)

    Non-Fiction:
    - Sapiens: A Brief History of Humankind (Yuval Noah Harari)
    - Educated (Tara Westover)
    - Atomic Habits (James Clear)
    - The Psychology of Money (Morgan Housel)

    Self-Development:
    - Deep Work (Cal Newport)
    - The 7 Habits of Highly Effective People (Stephen R. Covey)
    - Think and Grow Rich (Napoleon Hill)
    - Ikigai: The Japanese Secret to a Long and Happy Life (Héctor García & Francesc Miralles)

    Technology & Data:
    - Artificial Intelligence: A Guide for Thinking Humans (Melanie Mitchell)
    - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)
    - Data Science for Business (Foster Provost & Tom Fawcett)
    - Python Crash Course (Eric Matthes)

    Children's Books:
    - Harry Potter and the Sorcerer's Stone (J.K. Rowling)
    - Charlie and the Chocolate Factory (Roald Dahl)
    - The Little Prince (Antoine de Saint-Exupéry)
    - Diary of a Wimpy Kid (Jeff Kinney)

    Notes:
    - Some books may be temporarily out of stock.
    - Prices vary depending on edition (paperback, hardcover, or e-book).
    - Special requests: Gift wrapping available upon request.
    """

@tool
def add_to_order(book: str, options: Iterable[str] = None) -> str:
    """Adds the specified book to the customer's order, including any options 
    (e.g., edition type like hardcover, paperback, or e-book)."""
    pass

@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct."""
    pass

@tool
def get_order() -> str:
    """Returns the user's order so far. One item per line."""
    pass

@tool
def clear_order():
    """Removes all items from the user's order."""
    pass

@tool
def place_order() -> int:
    """Sends the order to the bookstore for fulfillment."""
    pass

# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_catalog]
tool_node = ToolNode(auto_tools)

# Order tools will be handled by the order node
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# The LLM needs to know about all tools
llm_with_tools = llm.bind_tools(auto_tools + order_tools)

def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools."""
    message_history = [BOOKSTOREBOT_SYSINT] + state["messages"]
    response = llm_with_tools.invoke(message_history)
    return {"messages": [response]}

def order_node(state: OrderState) -> OrderState:
    """The ordering node. This is where the order state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":
            options = tool_call["args"].get("options", [])
            option_str = ", ".join(options) if options else "standard edition"
            order.append(f'{tool_call["args"]["book"]} ({option_str})')
            response = f"Added to order: {tool_call['args']['book']}"

        elif tool_call["name"] == "confirm_order":
            if not order:
                response = "Your order is currently empty."
            else:
                order_text = "\n".join([f"  - {book}" for book in order])
                response = f"Your current order:\n{order_text}\n\nIs this correct?"

        elif tool_call["name"] == "get_order":
            response = "\n".join(order) if order else "(no order)"

        elif tool_call["name"] == "clear_order":
            order.clear()
            response = "Order cleared."

        elif tool_call["name"] == "place_order":
            if not order:
                response = "Cannot place empty order."
            else:
                order_text = "\n".join(order)
                print(f"Sending order to bookstore system!\n{order_text}")
                order_placed = True
                response = f"Order placed successfully! Estimated delivery: {randint(1, 7)} days."

        else:
            response = f"Unknown tool call: {tool_call['name']}"

        outbound_msgs.append(
            ToolMessage(
                content=str(response),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}

def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if state.get("finished", False):
        return END
    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Check if it's an auto tool or order tool
        for tool_call in msg.tool_calls:
            if tool_call["name"] in [tool.name for tool in auto_tools]:
                return "tools"
            elif tool_call["name"] in [tool.name for tool in order_tools]:
                return "ordering"
        return "tools"  # default fallback
    else:
        return END

# Build the graph
graph_builder = StateGraph(OrderState)

# Add nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

# Compile the graph
graph_with_order_tools = graph_builder.compile()

def chatbot_response(user_input: str) -> str:
    """Main function to get chatbot response."""
    try:
        # Initialize state properly
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "order": [],
            "finished": False
        }
        
        # Invoke the graph
        result = graph_with_order_tools.invoke(initial_state)
        
        # Get the last assistant message
        messages = result.get("messages", [])
        if messages:
            # Find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'content') and isinstance(msg, AIMessage):
                    return msg.content
            # If no AI message found, return the last message content
            return str(messages[-1].content) if messages[-1].content else "Maaf, terjadi kesalahan."
        
        return "Maaf, tidak ada respons yang tersedia."
        
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {str(e)}"
