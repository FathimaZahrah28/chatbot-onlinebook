import google.generativeai as genai
import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage

my_api_key = "AIzaSyCmS4s2HfasXfVHwLpG0PFnZ0B5NESOxB0"
os.environ["GOOGLE_API_KEY"] = my_api_key
genai.configure(api_key=my_api_key)




class OrderState(TypedDict):
    """State representing the customer's order conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress order.
    order: list[str]

    # Flag indicating that the order is placed and completed.
    finished: bool


# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
BARISTABOT_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
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
    "say goodbye!",

)

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the BookStoreBot. Type `q` to quit. How can I help you find a book today?"


# Try using different models. The `pro` models perform the best, especially
# with tool-calling. The `flash` models are super fast, and are a good choice
# if you need to use the higher free-tier quota.
# Check out the features and quota differences here: https://ai.google.dev/pricing
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


def chatbot(state: OrderState) -> OrderState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    message_history = [BARISTABOT_SYSINT] + state["messages"]
    return {"messages": [llm.invoke(message_history)]}


# Set up the initial graph based on our state definition.
graph_builder = StateGraph(OrderState)

# Add the chatbot function to the app graph as a node called "chatbot".
graph_builder.add_node("chatbot", chatbot)

# Define the chatbot node as the app entrypoint.
graph_builder.add_edge(START, "chatbot")

chat_graph = graph_builder.compile()

def human_node(state: OrderState) -> OrderState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}


def chatbot_with_welcome_msg(state: OrderState) -> OrderState:
    """The chatbot itself. A wrapper around the model's own chat interface."""

    if state["messages"]:
        # If there are messages, continue the conversation with the Gemini model.
        new_output = llm.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        # If there are no messages, start with the welcome message.
        new_output = AIMessage(content=WELCOME_MSG)

    return state | {"messages": [new_output]}


# Start building a new graph.
graph_builder = StateGraph(OrderState)

# Add the chatbot and human nodes to the app graph.
graph_builder.add_node("chatbot", chatbot_with_welcome_msg)
graph_builder.add_node("human", human_node)

# Start with the chatbot again.
graph_builder.add_edge(START, "chatbot")

# The chatbot will always go to the human next.
graph_builder.add_edge("chatbot", "human");

from typing import Literal


def maybe_exit_human_node(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"


graph_builder.add_conditional_edges("human", maybe_exit_human_node)

chat_with_human_graph = graph_builder.compile()

from langchain_core.tools import tool


@tool
def get_catalog() -> str:
    """Provide the latest up-to-date book catalog."""
    # Note: This is just hard-coded text, but you could connect this to a live stock
    # database, or you could even integrate with an external API from a bookstore/publisher.

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
    - Harry Potter and the Sorcerer’s Stone (J.K. Rowling)
    - Charlie and the Chocolate Factory (Roald Dahl)
    - The Little Prince (Antoine de Saint-Exupéry)
    - Diary of a Wimpy Kid (Jeff Kinney)

    Notes:
    - Some books may be temporarily out of stock.
    - Prices vary depending on edition (paperback, hardcover, or e-book).
    - Special requests: Gift wrapping available upon request.
  """

from langgraph.prebuilt import ToolNode


# Define the tools and create a "tools" node.
tools = [get_catalog]
tool_node = ToolNode(tools)

# Attach the tools to the model so that it knows what it can call.
llm_with_tools = llm.bind_tools(tools)


def maybe_route_to_tools(state: OrderState) -> Literal["tools", "human"]:
    """Route between human or tool nodes, depending if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    # Only route based on the last message.
    msg = msgs[-1]

    # When the chatbot returns tool_calls, route to the "tools" node.
    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        return "tools"
    else:
        return "human"


def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([BARISTABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}


graph_builder = StateGraph(OrderState)

# Add the nodes, including the new tool_node.
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)

# Chatbot may go to tools, or human.
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human may go back to chatbot, or exit.
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_menu = graph_builder.compile()

from collections.abc import Iterable
from random import randint

from langgraph.prebuilt import InjectedState
from langchain_core.messages.tool import ToolMessage

# These functions have no body; LangGraph does not allow @tools to update
# the conversation state, so you will implement a separate node to handle
# state updates. Using @tools is still very convenient for defining the tool
# schema, so empty functions have been defined that will be bound to the LLM
# but their implementation is deferred to the order_node.


@tool
def add_to_order(book: str, options: Iterable[str]) -> str:
    """Adds the specified book to the customer's order, including any options 
    (e.g., edition type like hardcover, paperback, or e-book).

    Returns:
      The updated order in progress.
    """


@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct.

    Returns:
      The user's free-text response.
    """


@tool
def get_order() -> str:
    """Returns the user's order so far. One item per line."""


@tool
def clear_order():
    """Removes all items from the user's order."""


@tool
def place_order() -> int:
    """Sends the order to the bookstore for fulfillment.

    Returns:
      The estimated number of days until the order is delivered (or available for pickup).
    """


def order_node(state: OrderState) -> OrderState:
    """The ordering node. This is where the order state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:

        if tool_call["name"] == "add_to_order":

            # Each order item is just a string. This is where it is assembled as "book (options, ...)".
            options = tool_call["args"]["options"]
            option_str = ", ".join(options) if options else "no options"

            order.append(f'{tool_call["args"]["book"]} ({option_str})')
            response = "\n".join(order)

        elif tool_call["name"] == "confirm_order":

            # It's a good practice to show the user the exact data that comprises their order
            # so that what they confirm precisely matches the order that goes to the bookstore system.

            print("Your current order:")
            if not order:
                print("  (no items)")

            for book in order:
                print(f"  {book}")

            response = input("Is this correct? ")

        elif tool_call["name"] == "get_order":

            response = "\n".join(order) if order else "(no order)"

        elif tool_call["name"] == "clear_order":

            order.clear()
            response = None

        elif tool_call["name"] == "place_order":

            order_text = "\n".join(order)
            print("Sending order to bookstore system!")
            print(order_text)

            # TODO(you!): Implement bookstore fulfillment.
            order_placed = True
            response = randint(1, 7)  # ETA in days for delivery or pickup

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
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
        # When a book order is placed, exit the app. 
        # The system instruction indicates that the chatbot should 
        # say thanks and goodbye at this point, so we can exit cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "ordering"

    else:
        return "human"
    
# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_catalog]   # was get_menu
tool_node = ToolNode(auto_tools)

# Book-order tools will be handled by the order node.
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# The LLM needs to know about all of the tools, so specify everything here.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


graph_builder = StateGraph(OrderState)

# Nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Chatbot -> {ordering, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools (both kinds) always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_order_tools = graph_builder.compile()

config = {"recursion_limit": 100}

# Uncomment this line to execute the graph:
state = graph_with_order_tools.invoke({"messages": []}, config)