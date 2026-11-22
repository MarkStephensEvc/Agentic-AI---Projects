import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Union, Any
from sqlalchemy import create_engine, Engine
from pathlib import Path
import json
import re

# Agentic framework
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Create an SQLite database
db_engine = create_engine("sqlite:///beavers_choice.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Beaver's Choice Paper Company database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(PATH / "project" / "quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(PATH / "project" / "quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]
######################################################
# Additional helper functions
def normalise_iso_date(raw: str) -> str:
    """Validate/coerce a date string to YYYY-MM-DD. Returns '' if unparseable."""
    if not raw:
        return ""
    raw = raw.strip()
    # Try strict ISO first (the format we asked the LLM for)
    try:
        return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    # Fallbacks for common LLM/customer slips
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return "" 

def get_min_stock_levels() -> Dict[str, int]:
    """Return each item's reorder threshold from the inventory reference table."""
    df = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)
    return dict(zip(df["item_name"], df["min_stock_level"]))


def get_unit_pricing() -> Dict[str, int]:
    """Return each item's unit_pricing in inventory table."""
    df = pd.read_sql("SELECT item_name, unit_price FROM inventory", db_engine)
    return dict(zip(df["item_name"], df["unit_price"]))
 
def parse_agent_result(raw) -> dict:
    """Normalise a sub-agent's return (dict or JSON string) into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}
    return {} 

def get_price(item_qty: Dict) -> Dict:
    """Compute the base price for an order: per-item line totals, subtotal, and a
    bulk discount applied to large orders.

     A 15% bulk discount applies to large item orders more than 5000,
     while a 10% bulk discount applies to large item orders more than 1000.
     The total amount is rounded to the nearest tens.
      
    Args:
        item_qty: Mapping of inventory item names to quantities.

    Returns:
        A dict with per-item price totals, per-item discount rates applied,
        and total amount.
    """
    pricing = get_unit_pricing()                # {item_name: unit_price} from inventory table
    item_price = {}
    discount_rates={}
    total = 0

    for name, qty in item_qty.items():
        qty = int(re.sub(r"[^\d]", "", str(qty)) or 0)
        if name not in pricing:
            continue 
        if qty > 5000:
            discount = 0.15
        elif qty > 1000:
            discount = 0.10
        else:
            discount = 0
        item_total = (1 - discount) * pricing[name] * qty
        item_price[name] = item_total 
        discount_rates[name] = f'{discount*100}%'
        total += item_total
        
    total = round(total,-1) # round to the nearest tens.
    
    return {
        "item_price": item_price,
        "discount_rates": discount_rates,
        "total_amount": total
    }    

def to_int(v) -> int:
    """Coerce an agent-supplied value to int; non-numeric -> 0."""
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        digits = re.sub(r"[^\d]", "", v)   # "200 sheets" -> "200"
        return int(digits) if digits else 0
    return 0   # dicts, lists, None -> 0


def stock_delivery(item_name: str, as_of_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date for a restock of a given size, if the
    item is one the company's suppliers carry.

    Args:
        item_name: The catalogue item to restock.
        as_of_date: ISO date (YYYY-MM-DD) the restock would be placed.
        quantity: Units to reorder.

    Returns:
        The estimated delivery date as ISO (YYYY-MM-DD), or the string
        "UNAVAILABLE" if no supplier carries the item.
    """
    catalogue = {p["item_name"] for p in paper_supplies}
    #Case-insensitive exact match
    lower = {c.lower(): c for c in catalogue}
    if item_name.lower() in lower:
        return get_supplier_delivery_date(as_of_date, quantity)
    if item_name not in catalogue:
        return "UNAVAILABLE"
    return get_supplier_delivery_date(as_of_date, quantity)
   
########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# Set up and load your env parameters and instantiate your model.

PATH = Path(os.getcwd())
dotenv.load_dotenv(PATH / "config.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Paper Order State Management
@dataclass
class PaperOrder:
    request: str
    item_qty: Dict[str, Any]
    event: str
    order_type: str = ""
    request_date: str = ""
    required_date: str = ""
    stock_delivery_date: str = ""
    reorder: Dict[str, Any] = field(default_factory=dict)
    deadline_met: bool = True
    total_amount: int = 0               
    quote_explanation: str = ""
    transaction_ids: List[Any] = field(default_factory=list)     

@dataclass
class SalesState:
    order: PaperOrder = None

sales_state = SalesState()   

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Tools for inventory agent
@tool
def check_inventory(as_of_date: str) -> Dict[str, int]:
    """Return current stock as of a date for every item that has positive stock. 
    This is the set of official catalogue names to match the customer's requested items against; 
    an item missing from this list means zero stock, not an invalid item.
    
    Args:
        as_of_date (str): ISO date (YYYY-MM-DD) to compute stock as of.

    Returns:
        Dict[str, int]: A dictionary of paper items and quantity currently in stock.
    """
    
    return get_all_inventory(as_of_date)
   
# Tools for quoting agent
@tool
def get_quote_history(event: str, item_names: List[str]) -> List[Dict]:
    """Find comparable past quotes to guide pricing, relaxing the search until
    enough matches are found.

    Builds search terms from the order's event and item names, then queries
    history. Because terms are AND-matched, an over-specific search can return
    too few rows; this drops the least-important term and retries until at
    least 3 quotes are found or only the event term remains.

    Args:
        event: The event the order is for (e.g. "ceremony").
        item_names: The canonical item names in this order.

    Returns:
        Up to 5 past quote records (original_request, total_amount,
        quote_explanation, job_type, order_size, event_type, order_date).
    """
    # Most-specific first:  each item name, then event.
        
    terms = []
    if event:
        terms.append(event.strip().lower())
    for name in item_names:
        t = name.strip().lower()
        if t and t not in terms:
            terms.append(t)
    
    # query, and if too few rows, drop the last
    # (least-important) term and retry.
    results = []
    while terms:
        results = search_quote_history(terms, limit=5)
        if len(results) >= 3:
            break
        terms = terms[:-1]   # drop the least-important remaining term

    return results   
    
@tool
def record_transaction(item_name: str, transaction_type: str, quantity: int,
                       price: float, transaction_date: str) -> int:
    """Record a single transaction (one sale or one restock) to the transaction record.

    Args:
        item_name: The catalogue item being sold or restocked.
        transaction_type: 'sales' or 'stock_orders'.
        quantity: Units in this entry.
        price: Total price for this entry (not per-unit).
        transaction_date: ISO date (YYYY-MM-DD) to record the entry under. Sales are dated
            on the request date; restocks on their delivery date.

    Returns:
        The ID of the newly recorded transaction.
    """
    return create_transaction(item_name, transaction_type,  quantity , price , transaction_date) 
    
    
@tool
def cash_check(as_of_date: str) -> float:
    """Report the company's available cash balance as of a date.

    This is informational only — it does not block any transaction. Use it to
    state available funds when recording or explaining an order.

    Args:
        as_of_date: ISO date (YYYY-MM-DD) to compute the balance as of.

    Returns:
        The cash balance as of that date (may be negative).
    """
    return round(get_cash_balance(as_of_date), 2)  

       


# Set up your agents and create an orchestration agent that will manage them.
class InventoryQueryAgent(ToolCallingAgent):
    """Agent responsible for matching requested items to official inventory names
    and reporting current stock levels."""
    def __init__(self, model):
        super().__init__(
            tools=[check_inventory],
            model=model,
            name="inventory_query_agent",
            description="""
            You are an inventory specialist for Beaver's Choice Paper Company.
            Your only job is to identify, for each requested item, its official
            inventory name and how many units are currently in stock.

            Your method, always:
            - Use check_inventory to read the official item names and the quantities
              currently in stock.
            - Match each requested item to its closest official name in the inventory 
             (from check_inventory). Match on the meaningful keywords (excluding 'paper') 
              and be as precise as possible; consider each match carefully before
              settling on it.
            - An item not found in current inventory simply has 0 on-hand stock — it
              is NOT invalid and must NOT be dropped. Keep it under the customer's
              wording with a stock of 0.
            - Report stock figures exactly as read from check_inventory; never
              estimate, round, or invent numbers.
            - Do NOT compute shortfalls, restock quantities, or delivery dates, and
              do not decide availability — those are handled after you return.
        """
        )

class QuotationAgent(ToolCallingAgent):
    """Agent responsible for generating prices and considering discounts"""
    def __init__(self, model):
        super().__init__(
            tools=[get_quote_history],
            model=model,
            name="quotation_agent",
            description="""
            You are an expert in providing quotations for paper products. You price
            customer paper orders for Beaver's Choice Paper Company.

            Your method, always:
            - Use get_quote_history as your PRIMARY guide: study comparable past
              quotes to judge a fair, competitive price and whether a bulk discount per item order
              is warranted. Prioritise your judgement.
            - Use the baseline price provided to you as CONTEXT for your judgement,so you can see
              how your history-based price compares to the underlying numbers, but favour your history-based price.  
              If get_quote_history returns nothing, use the get_price total directly.
            - Write explanations in the same warm, customer-facing tone as past
              quotes (returned by get_quote_history), mentioning a discount ONLY
              when you actually applied one.
        """,
        )

class TransactionAgent(ToolCallingAgent):
    """Agent responsible for carrying out transactions, either for sale or restock orders."""
    def __init__(self, model):
        super().__init__(
            tools=[cash_check, record_transaction ],
            model=model,
            name="transaction_agent",
            description="""
            You are the transactions specialist for Beaver's Choice Paper Company.
            You record finalised orders to the transaction record; you do not re-decide pricing,
            stock, or whether to proceed — those are already determined when an order reaches you.

            Your method, always:
            - Use cash_check to report the company's available funds before recording
              a purchase (a stock order spends money). This is for visibility only: 
              still record the order even if funds are low — the balance is allowed to 
              go negative. 
            - Use record_transaction to write each entry exactly as instructed:
              a 'sales' entry for a customer order, and a 'stock_orders' entry for
              any restock. A single order may require BOTH.
            - Record sales dated on the request date, and restocks dated on their
              delivery date, so stock is counted only once it arrives.
            - Report the resulting transaction IDs. Do not invent amounts, dates,
              or quantities — use only the values provided in the order.
        """,
        )
                

class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates the multi-agent Sales Team for Beaver's Choice Paper Company."""
    
    def __init__(self, model):
        self.model = model
        
        # Initialize specialized agents
        self.Inventory_agent = InventoryQueryAgent(model)
        self.quotation_agent = QuotationAgent(model)
        self.transaction_agent = TransactionAgent(model)
       
        @tool
        def inventory_management(item_qty: dict, event: str,
                                 request_date: str, required_date: str) -> str:
            """Delegate stock matching to the Inventory Query Agent, then compute
            shortfalls, restock delivery dates, and availability in Python.
            Updates the shared order. Returns a status summary.

            Args:
                item_qty: Requested items in the customer's OWN wording -> quantity.
                event: The customer's event (e.g. "ceremony").
                request_date: ISO date the request was made (YYYY-MM-DD).
                required_date: ISO delivery deadline (YYYY-MM-DD), or "" if none.

            Returns:
                A status string stating order_type, deadline_met, and any unavailable items.
            """
            
            ord = sales_state.order
            
            request_date  = normalise_iso_date(request_date)
            required_date = normalise_iso_date(required_date)
                        
            #Sub-agent matches items + reads live stock
            raw  = self.Inventory_agent.run(f"""
            Match this customer order to official inventory and report stock.

            Items requested (customer wording -> quantity): {item_qty}
            As-of date for stock: {request_date}

            1. Call check_inventory  EXACTLY ONCE with as_of_date = {request_date}.
               Use the result for all items; do not call it again.
            2. Consider all official inventory item names returned by check_inventory and the
               requested item names in lowercase. Match each requested item to its closest
               official inventory name. The official name is usually the requested phrase with
               size/colour/quality modifiers removed. Examples:
                - "A4 glossy paper"            -> "Glossy paper"     (drop the "A4")
                - "heavy cardstock (white)"    -> "Cardstock"        (drop "heavy", "(white)")
                - "colored paper (assorted colors)" -> "Colored paper"  (drop "(assorted colors)")
                - "sheets of colorful poster paper" -> "Poster paper" (drop "sheets of", "colorful")
                - "A4_paper" / "A4 printer paper"   -> "A4 paper"     (no underscores; drop "printer")
                - "white cardstock"            -> "Cardstock"
                - "rolls of crepe paper"       -> "Crepe paper"       (drop "rolls of")
                - "balloons"                   -> no match -> keep "balloons", stock 0
                - "A3 paper"                   -> no match -> keep "A3 paper", stock 0
                - "printer reams"              -> A4 paper
               If an item has no reasonable match in current inventory, KEEP it under the
               customer's wording with a stock of 0 — do not drop it and do not force a
               wrong match (e.g. do NOT map "poster paper" to "Crepe paper").
            3. Include EVERY requested item exactly once, with the SAME quantity given above —
               never change, split, or drop a requested quantity.
            4. Return EXACTLY ONE final answer: a single JSON object and call final_answer only once.
               Both dicts keyed by the same matched name (prefer the official inventory name):
               "matched": {{'matched name': requested_quantity}},
               "stock":   {{'matched name': units_in_stock}}
            Do NOT compute shortfalls, just match and report stock. Do not include 
            underscores in the item names.
            """)
            
            result = parse_agent_result(raw)
            raw_matched = result.get("matched", {})
            raw_stock   = result.get("stock", {})
            mins = get_min_stock_levels()
                                 
            #parse requested_item and stock to string only keys and integer valued entries
            matched = {str(k): to_int(v) for k, v in raw_matched.items()} if isinstance(raw_matched, dict) else {}
            stock   = {str(k): to_int(v) for k, v in raw_stock.items()}   if isinstance(raw_stock, dict)   else {}

            # deadline only fails when a date was given AND restock arrives after it
            # parse maximum delivery date bundling all deliveries to the one day (latest day)
            reorder = {}
            has_shortfall = False
            delivery_dates = []
            unavailable_items = []

            for name, qty in matched.items():
                on_hand = stock.get(name, 0)
                shortfall = qty - on_hand

                if shortfall > 0:                       # ONLY short items hit stock_delivery
                    has_shortfall = True
                    verdict = stock_delivery(name, request_date, shortfall)
                    if verdict == "UNAVAILABLE":
                        unavailable_items.append(name)
                    else:
                        delivery_dates.append(verdict)

                # reorder to restore minimum stock level (only for catalogue items)
                projected = on_hand - qty
                min_level = mins.get(name, 0)
                if projected < min_level and name not in unavailable_items:
                    reorder[name] = min_level - projected

            delivery_date = max(delivery_dates) if delivery_dates else ""
            
            
            if unavailable_items:
                order_type = "stock_orders"
                deadline_met = False  
            elif not has_shortfall:
                order_type = "sales"          # sellable from stock now
                deadline_met = True
            else:
                order_type = "stock_orders"         # short -> record the restock regardless    
                deadline_met = not (required_date  and delivery_date and delivery_date > required_date)

            # Update PaperOrder
            ord.item_qty            = matched
            ord.event               = event
            ord.order_type          = order_type
            ord.request_date        = request_date
            ord.required_date       = required_date
            ord.stock_delivery_date = delivery_date
            ord.reorder             = reorder
            ord.deadline_met        = deadline_met
            ord.unavailable_items   = unavailable_items 
            
            return (f"Inventory assessed. order_type={order_type}, "
                    f"deadline_met={deadline_met}, unavailable_items={unavailable_items}, "
                    f"matched_items={list(matched.keys())}.")
                
            
        # Orchestrator tools for calling the other specialist Agents
        @tool
        def quote_generation() -> Dict:
            """Delegate pricing and quote-explanation to the Quotation Agent.
            Determine a base price on current order. Have the Quotation Agent 
            judge a final total and write the explanation, then update the order state. 
            Returns a short status summary.

            Returns:
                A status string stating the quoted total.
            """
            #current order state
            ord = sales_state.order
            
            if not ord.item_qty:
                return "Cannot quote: inventory not yet assessed. Call inventory_management first, then quote_generation."
            
            # baseline pricing
            base = get_price(ord.item_qty)
            baseline_total = base["total_amount"]
            
            # The agent judges a final price from baseline + comparable history
            raw = self.quotation_agent.run(f"""
                Decide ONE final price for a "{ord.event}" event order.

                Baseline price (already computed for you): {baseline_total}
                Items being priced: {ord.item_qty}

                Steps:
                1. Call get_quote_history ONCE with the event "{ord.event}" and these
                   item names: {list(ord.item_qty.keys())}.
                2. Read the comparable past quotes it returns. Lookt at 'total_amount' to consider the quote
                   and look at 'quote_explanation' to consider the style of the customer reply and whether discounts 
                   were issued. They are REFERENCE ONLY —  do not return them, and do not produce one answer per quote.
                3. Decide a SINGLE final total_amount for THIS order, guided by that history
                   and the baseline price above, leaning to the larger number. If history is empty, use the baseline. 
                4. Write a 2-4 sentence quote_explanation for THIS order, similar to the style from comparable past quotes
                   mentioning any discounts.

                Return EXACTLY ONE final answer: a single JSON object, and call final_answer
                only once.
                "total_amount": one number rounded to nearest ten (no '$' or 'dollars'),
                "quote_explanation": string
            """)
            
            result = parse_agent_result(raw)
            
            if isinstance(result, list):
                result = sum(result)/len(result) if result else {}      # take the last answer if it sprayed several
            if isinstance(result, (int, float, str)):
                result = {"total_amount": result}

            # Coerce the total separately — a bad number falls back, doesn't lose the explanation
            try:
                total = int(round(float(result.get("total_amount", 0.0))))
            except (TypeError, ValueError):
                total = 0
                           
            if not total or total <= 0:                         
                total = baseline_total   #fallback from get_price
            
            # catch a quote of zero
            if ord.item_qty and total <= 0:
                total = baseline_total
                
            ord.total_amount      = int(round(total))
            ord.quote_explanation = result.get("quote_explanation", "")
    
            return f"Quote ready. total_amount={ord.total_amount}."
        
        @tool
        def order_fulfillment() -> str:
            """Delegate transaction recording to the Transaction Agent.

            Record the current order's transactions (the sale and/or any restock)
            to the transaction record, updating the shared order with the resulting
            transaction IDs. 
            Returns a short status summary.

            Args:
                order_details: The order dict (PaperOrder schema) from quote_generation
                    or inventory_management.

            Returns:
                A status string stating how many transactions were recorded.
            """
            #current order state
            ord = sales_state.order
                       
            pricing = get_unit_pricing()
            entries = []
            
            # Successful Sales
            if ord.order_type == "sales" and ord.deadline_met:
                # Per-item base cost, to split the quoted total proportionally
                base = {name: pricing.get(name, 0.0) * qty for name, qty in ord.item_qty.items()}
                base_subtotal = sum(base.values())
                items = list(ord.item_qty.items())
                allocated = 0.0
                # Determine the price allocation per item under the quotation arrived at.
                for i, (name, qty) in enumerate(items):
                    if i < len(items) - 1 and base_subtotal > 0:
                        share = round(ord.total_amount * base[name] / base_subtotal, 2)
                    else:
                        # last item absorbs the remainder so prices sum to total_amount exactly
                        share = round(ord.total_amount - allocated, 2)
                    allocated += share
                    entries.append({
                        "item_name": name,
                        "transaction_type": "sales",
                        "quantity": int(qty),
                        "price": share,
                        "transaction_date": ord.request_date,
                    })
                    
            # Restock: one row per reorder item, at cost, dated on arrival
            for name, qty in ord.reorder.items():
                entries.append({
                    "item_name": name,
                    "transaction_type": "stock_orders",
                    "quantity": int(qty),
                    "price": round(pricing.get(name, 0.0) * int(qty), 2),
                    "transaction_date": ord.stock_delivery_date or ord.request_date,
                })
            
            if not entries:
                # Nothing to record (e.g. declined order with no restock needed).
                ord.transaction_ids = []
                return "No transactions to record."
            
            # --- Agent records each pre-built entry and reports cash ---
            raw = self.transaction_agent.run(f"""
                Record these finalised ledger entries for the order.

                Cash check date: {ord.request_date}
                Entries to record (record EACH one exactly as given): {entries}

                1. Call cash_check with {ord.request_date} to report available funds (informational only).
                2. If the entries list is empty, record nothing and return an empty list.
                3. Otherwise, for EACH entry call record_transaction with its item_name, transaction_type,
                  quantity, price, and transaction_date EXACTLY as given. Never call record_transaction with 
                  empty or zero values, and never invent an entry not in the list.
                3. Return ONLY a JSON object:
                "transaction_ids": [ids returned by record_transaction],
            """)
            
            
            result = parse_agent_result(raw)

            # Write results back onto the state object
            ord.transaction_ids = result.get("transaction_ids", [])
            return  f"Recorded {len(ord.transaction_ids)} transaction(s)."
        
        @tool
        def get_order_summary() -> dict:
            """Return the current order's full details, for composing the final customer message. 
            Call this AFTER fulfilment, before writing the reply.

            Returns:
                The current order as a dict (item_qty, event, order_type, dates,
                total_amount, quote_explanation, transaction_ids, deadline_met, etc.).
            """
            return asdict(sales_state.order)
          

        super().__init__(
                    tools=[inventory_management, quote_generation, order_fulfillment, get_order_summary],
                    model=model,
                    name="orchestrator",
                    description="""
                    You are the orchestrator for Beaver's Choice Paper Company.
                    You coordinate the inventory query agent, quotation agent, and transaction agent to handle
                    a customer order. Your job is to call the tools in the right sequence and write the final customer message.
                    
                    Call ONE tool at a time. Wait for each tool's result before deciding the next call. Never call quote_generation
                    or order_fulfillment in the same step as inventory_management.
                    
                    Workflow for each request:
                                    
                    1. Extract from the request:
                        - item_qty: requested items and quantities (as an integer), in the customer's OWN wording.
                          Item names must be plain phrases with spaces, exactly as the customer wrote them (do not guess catalogue names;
                          the inventory agent resolves those);
                        - event: the event mentioned (e.g. "ceremony");
                        - required_date: requested delivery date as YYYY-MM-DD, or "" if none stated
                            (do not invent one).
                            
                    2. Call inventory_management with those fields. Read the status string for "order_type"
                       and "deadline_met".
                    
                    3. Decide based on those two facts: 
                        - If deadline_met is FALSE: the order is DECLINED - it was NOT placed and NOT confirmed. Do NOT call quote_generation.
                          Call order_fulfillment once (it will record only the restock, not a sale). Then go to step 5
                          and decline.
                        - If order_type is "sales": the order is SUCCESSFUL. Call quote_generation, then order_fulfillment.
                        - If order_type is "stock_orders" and deadline_met is TRUE: the order is POSTPONED - the sale did NOT occur. Call order_fulfillment 
                          to record the restock; the order will be fulfilled once the restock arrives.   
                                              
                    4. Call get_order_summary to retrieve the order's final details.
                    
                    5. Write ONE clear, friendly customer message in plain prose:
                        - DECLINED order (deadline_met FALSE): Do NOT say "confirmed", "placed", "processed", or give a total or transaction reference.
                          Apologise, state plainly that we were unable to place the order, and name the specific item(s) that are unavailable (from
                          unavailable_items) as the reason. Invite them to adjust the order.
                          Example shape: "Unfortunately we couldn't place your order because [item] isn't something we're able to supply. We're sorry for the
                          inconvenience — if you'd like to adjust the order, we're happy to help."
                        - POSTPONED order (order_type "stock_orders" and deadline_met TRUE): the sale has NOT been completed and the customer has NOT been charged. Do NOT
                          give a total_amount or a transaction reference, and do NOT say the order is "confirmed", "purchased", or "successfully placed". 
                          Explain that some items aren't currently in stock but CAN be restocked, give the expected restock delivery date, 
                          and that it arrives before their required date. Invite them to confirm once the stock is in. 
                          Example shape: "Thanks for your order. We don't currently have enough [item] in stock to fulfil it right away, but we can restock and expect it
                          by [delivery date], ahead of your [required date] deadline. We'll be in touch to finalise once it arrives — just let us know if you'd like to go
                          ahead."
                        - SUCCESSFUL order (order_type "sales"): give a warm summary including the quote explanation (mention any discount and which items, but NOT how the
                          quote was determined), the total_amount, and the transaction reference. If any item needed a restock, note the expected delivery date and that it
                          arrives before the required date.
                          
                    Never mention internal field names (order_type, deadline_met, reorder) or tool names in the customer message — 
                    write naturally, as a sales rep would.
                    """,
                )
        
    def process_order(self, request_with_date: str) -> str:
        """
        Process a customer order from initial request through transaction.
        
        Args:
            request_with_date: Natural language order request from customer and date of request
            
        Returns:
            Response to customer with order details and status
        """
        marker = " (Date of request: "

        customer_request, date_part = request_with_date.rsplit(marker, 1)
        request_date = date_part[:-1]  # Remove trailing ')'
           
        # Fresh order for this request — attribute mutation, no global needed
        sales_state.order = PaperOrder(request=customer_request,item_qty={}, event="", request_date=request_date) 
              
        context = f"""
            Customer request: "{customer_request}"
            Request date (YYYY-MM-DD): {request_date}

            Handle this request using your workflow.
        """
        return self.run(context)
# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    file_path = PATH / "project"
    print(file_path)
    try:
        quote_requests_sample = pd.read_csv(file_path / "quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    
    orchestrator = Orchestrator(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = orchestrator.process_order(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv(file_path / "test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
