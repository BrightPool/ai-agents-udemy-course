from typing import List

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """Line item for the chatbot."""

    description: str = Field(description="The description of the line item")
    quantity: int = Field(description="The quantity of the line item")
    price: float = Field(description="The price of the line item")
    unit_price: float = Field(description="The unit price of the line item")
    total: float = Field(description="The total price of the line item")


class InvoiceExtractionResult(BaseModel):
    """Result for the invoice extraction."""

    line_items: List[LineItem] = Field(description="The line items of the invoice")
    purchase_order: str = Field(description="The purchase order number")
    bill_to: str = Field(description="The bill to information")
    bill_to_address: str = Field(description="The bill to address")
    ship_to: str = Field(description="The ship to information")
    seller: str = Field(description="The seller information")
    subtotal_amount: str = Field(description="The subtotal amount")
    tax_amount: str = Field(description="The tax amount")
    total_amount: str = Field(description="The total amount")
    payment_terms: str = Field(description="The payment terms")
    notes: str = Field(description="Additional notes")
