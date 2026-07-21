import json
import re
import numpy as np
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ============================================================
# YOUR ACTUAL SCHEMA (directly from the markdown)
# ============================================================
SCHEMA = {
    "tables": [
        {
            "name": "customer",
            "description": "Stores registered customers and their account information.",
            "columns": [
                {"name": "customer_id", "pk": True, "fk": False, "ref": None},
                {"name": "cust_code", "pk": False, "fk": False, "ref": None},
                {"name": "full_name", "pk": False, "fk": False, "ref": None},
                {"name": "email", "pk": False, "fk": False, "ref": None},
                {"name": "phone", "pk": False, "fk": False, "ref": None},
                {"name": "status", "pk": False, "fk": False, "ref": None},
                {"name": "created_at", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "cust_addr",
            "description": "Stores customer shipping and billing addresses.",
            "columns": [
                {"name": "addr_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "addr_type", "pk": False, "fk": False, "ref": None},
                {"name": "line1", "pk": False, "fk": False, "ref": None},
                {"name": "city", "pk": False, "fk": False, "ref": None},
                {"name": "state", "pk": False, "fk": False, "ref": None},
                {"name": "postal_cd", "pk": False, "fk": False, "ref": None},
                {"name": "is_default", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "seller_mst",
            "description": "Stores marketplace sellers.",
            "columns": [
                {"name": "seller_id", "pk": True, "fk": False, "ref": None},
                {"name": "seller_nm", "pk": False, "fk": False, "ref": None},
                {"name": "email", "pk": False, "fk": False, "ref": None},
                {"name": "rating", "pk": False, "fk": False, "ref": None},
                {"name": "is_active", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "category",
            "description": "Product categories.",
            "columns": [
                {"name": "category_id", "pk": True, "fk": False, "ref": None},
                {"name": "category_name", "pk": False, "fk": False, "ref": None},
                {"name": "parent_category_id", "pk": False, "fk": True, "ref": {"table": "category", "col": "category_id"}}
            ]
        },
        {
            "name": "product",
            "description": "Master list of products sold on the platform.",
            "columns": [
                {"name": "product_id", "pk": True, "fk": False, "ref": None},
                {"name": "category_id", "pk": False, "fk": True, "ref": {"table": "category", "col": "category_id"}},
                {"name": "seller_id", "pk": False, "fk": True, "ref": {"table": "seller_mst", "col": "seller_id"}},
                {"name": "sku", "pk": False, "fk": False, "ref": None},
                {"name": "product_name", "pk": False, "fk": False, "ref": None},
                {"name": "brand", "pk": False, "fk": False, "ref": None},
                {"name": "is_active", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "inv_stock",
            "description": "Current inventory available in warehouses.",
            "columns": [
                {"name": "stock_id", "pk": True, "fk": False, "ref": None},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "warehouse_id", "pk": False, "fk": True, "ref": {"table": "warehouse", "col": "warehouse_id"}},
                {"name": "qty_available", "pk": False, "fk": False, "ref": None},
                {"name": "reserved_qty", "pk": False, "fk": False, "ref": None},
                {"name": "last_upd", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "warehouse",
            "description": "Warehouses storing inventory.",
            "columns": [
                {"name": "warehouse_id", "pk": True, "fk": False, "ref": None},
                {"name": "warehouse_name", "pk": False, "fk": False, "ref": None},
                {"name": "city", "pk": False, "fk": False, "ref": None},
                {"name": "state", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "tbl_ord_hdr",
            "description": "Customer order header.",
            "columns": [
                {"name": "ord_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "order_date", "pk": False, "fk": False, "ref": None},
                {"name": "sts_cd", "pk": False, "fk": False, "ref": None},
                {"name": "coupon_id", "pk": False, "fk": True, "ref": {"table": "coupon", "col": "coupon_id"}},
                {"name": "ship_addr_id", "pk": False, "fk": True, "ref": {"table": "cust_addr", "col": "addr_id"}},
                {"name": "total_amount", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "tbl_ord_item",
            "description": "Products belonging to an order.",
            "columns": [
                {"name": "ord_item_id", "pk": True, "fk": False, "ref": None},
                {"name": "ord_id", "pk": False, "fk": True, "ref": {"table": "tbl_ord_hdr", "col": "ord_id"}},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "qty", "pk": False, "fk": False, "ref": None},
                {"name": "unit_price", "pk": False, "fk": False, "ref": None},
                {"name": "discount_amt", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "pay_trn",
            "description": "Payment transactions for customer orders.",
            "columns": [
                {"name": "payment_id", "pk": True, "fk": False, "ref": None},
                {"name": "ord_id", "pk": False, "fk": True, "ref": {"table": "tbl_ord_hdr", "col": "ord_id"}},
                {"name": "payment_method", "pk": False, "fk": False, "ref": None},
                {"name": "payment_status", "pk": False, "fk": False, "ref": None},
                {"name": "txn_ref", "pk": False, "fk": False, "ref": None},
                {"name": "paid_amt", "pk": False, "fk": False, "ref": None},
                {"name": "paid_on", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "ship_hdr",
            "description": "Shipment details.",
            "columns": [
                {"name": "shipment_id", "pk": True, "fk": False, "ref": None},
                {"name": "ord_id", "pk": False, "fk": True, "ref": {"table": "tbl_ord_hdr", "col": "ord_id"}},
                {"name": "warehouse_id", "pk": False, "fk": True, "ref": {"table": "warehouse", "col": "warehouse_id"}},
                {"name": "carrier", "pk": False, "fk": False, "ref": None},
                {"name": "tracking_no", "pk": False, "fk": False, "ref": None},
                {"name": "shipped_date", "pk": False, "fk": False, "ref": None},
                {"name": "delivery_status", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "return_req",
            "description": "Customer return requests.",
            "columns": [
                {"name": "return_id", "pk": True, "fk": False, "ref": None},
                {"name": "ord_item_id", "pk": False, "fk": True, "ref": {"table": "tbl_ord_item", "col": "ord_item_id"}},
                {"name": "reason", "pk": False, "fk": False, "ref": None},
                {"name": "request_date", "pk": False, "fk": False, "ref": None},
                {"name": "return_status", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "rfnd_log",
            "description": "Refund processing records.",
            "columns": [
                {"name": "refund_id", "pk": True, "fk": False, "ref": None},
                {"name": "return_id", "pk": False, "fk": True, "ref": {"table": "return_req", "col": "return_id"}},
                {"name": "payment_id", "pk": False, "fk": True, "ref": {"table": "pay_trn", "col": "payment_id"}},
                {"name": "refund_amt", "pk": False, "fk": False, "ref": None},
                {"name": "refund_status", "pk": False, "fk": False, "ref": None},
                {"name": "processed_on", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "coupon",
            "description": "Coupons available to customers.",
            "columns": [
                {"name": "coupon_id", "pk": True, "fk": False, "ref": None},
                {"name": "coupon_code", "pk": False, "fk": False, "ref": None},
                {"name": "discount_type", "pk": False, "fk": False, "ref": None},
                {"name": "discount_value", "pk": False, "fk": False, "ref": None},
                {"name": "valid_from", "pk": False, "fk": False, "ref": None},
                {"name": "valid_to", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "review",
            "description": "Customer product reviews.",
            "columns": [
                {"name": "review_id", "pk": True, "fk": False, "ref": None},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "rating", "pk": False, "fk": False, "ref": None},
                {"name": "review_text", "pk": False, "fk": False, "ref": None},
                {"name": "reviewed_on", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "cart_item",
            "description": "Stores products currently added to a customer's shopping cart but not yet purchased.",
            "columns": [
                {"name": "cart_item_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "qty", "pk": False, "fk": False, "ref": None},
                {"name": "added_at", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "wishlist_item",
            "description": "Products customers have saved for future purchase.",
            "columns": [
                {"name": "wishlist_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "created_at", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "recently_viewed",
            "description": "Tracks products viewed by customers.",
            "columns": [
                {"name": "view_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "viewed_at", "pk": False, "fk": False, "ref": None}
            ]
        },
        {
            "name": "recommendation_log",
            "description": "Stores products recommended by the recommendation engine.",
            "columns": [
                {"name": "rec_id", "pk": True, "fk": False, "ref": None},
                {"name": "customer_id", "pk": False, "fk": True, "ref": {"table": "customer", "col": "customer_id"}},
                {"name": "product_id", "pk": False, "fk": True, "ref": {"table": "product", "col": "product_id"}},
                {"name": "algo_name", "pk": False, "fk": False, "ref": None},
                {"name": "score", "pk": False, "fk": False, "ref": None},
                {"name": "generated_at", "pk": False, "fk": False, "ref": None}
            ]
        }
    ]
}

# ============================================================
# RETRIEVER CLASS (same as before, uses SCHEMA)
# ============================================================
class TableRetriever:
    def __init__(self, schema: Dict[str, Any], embedding_model: str = "all-MiniLM-L6-v2"):
        self.schema = schema
        self.tables = schema["tables"]
        self.table_names = [t["name"] for t in self.tables]
        self.documents = []
        self.column_index = {}

        # NLP tools
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words("english"))
        except:
            self.stopwords = set()
        self.lemmatizer = WordNetLemmatizer()

        self._build_documents()
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.encoder = SentenceTransformer(embedding_model)
        self.table_embeddings = self.encoder.encode(self.documents, convert_to_tensor=True)

    def _build_documents(self):
        for table in self.tables:
            name = table["name"]
            desc = table.get("description", "")
            cols = [f"{col['name']} ({'PK' if col['pk'] else 'FK' if col['fk'] else ''})".strip()
                    for col in table["columns"]]
            pk_cols = [c["name"] for c in table["columns"] if c.get("pk")]
            fk_info = []
            for c in table["columns"]:
                if c.get("fk") and c.get("ref"):
                    fk_info.append(f"{c['name']} → {c['ref']['table']}.{c['ref']['col']}")

            # inverse relationships
            inverse = []
            for other in self.schema["tables"]:
                for c in other["columns"]:
                    if c.get("fk") and c.get("ref") and c["ref"]["table"] == name:
                        inverse.append(f"{other['name']}.{c['name']}")

            parts = [
                f"Table: {name}",
                f"Description: {desc}",
                f"Columns: {', '.join(cols)}",
                f"Primary Key: {', '.join(pk_cols)}",
                f"Foreign Keys: {', '.join(fk_info)}",
                f"Referenced by: {', '.join(inverse)}"
            ]
            doc = " ".join(parts)
            self.documents.append(doc)

            for col in table["columns"]:
                self.column_index[col["name"].lower()] = name

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stopwords and len(t) > 2]
        return tokens

    def retrieve(self, query: str, alpha: float = 0.5) -> List[str]:
        tokenized_q = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_q)

        query_emb = self.encoder.encode([query], convert_to_tensor=True)
        dense_scores = np.dot(self.table_embeddings, query_emb.T).flatten()

        bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
        if bm25_max > bm25_min:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.ones_like(bm25_scores)

        dense_norm = (dense_scores + 1) / 2

        final_scores = alpha * bm25_norm + (1 - alpha) * dense_norm

        # boost exact column/table name mentions
        query_lower = query.lower()
        for col, table in self.column_index.items():
            if col in query_lower:
                idx = self.table_names.index(table)
                final_scores[idx] += 0.2

        sorted_indices = np.argsort(final_scores)[::-1]
        return [self.table_names[i] for i in sorted_indices]



if __name__ == "__main__":
    retriever = TableRetriever(SCHEMA)

    # ====== EDIT THIS LIST: paste your 10 questions here ======
      # ====== Updated with your actual 11 questions ======
    EVALUATION_PROMPTS = [
        "Find customers who purchased products but never submitted a review.",           # Q1
        "Show all products sold by seller 'TechZone'.",                                 # Q2
        "List all orders where a coupon was applied during checkout.",                  # Q3
        "Find customers who have saved products to their wishlist.",                   # Q4
        "Show products that customers returned after they had already been delivered.", # Q5
        "Find customers who received refunds for returned products.",                   # Q6
        "List sellers whose products have received reviews with ratings below 2.",      # Q7
        "Find customers who added products to their cart but later purchased the same products.", # Q8
        "Identify products that were returned, refunded, and later purchased again by the same customer.", # Q9
        "Find customers who purchased products from different sellers, received shipments from multiple warehouses, returned at least one item, and whose refund has not yet been processed.", # Q10
        "Identify customers who viewed a product multiple times, added it to their wishlist, later purchased it using a coupon, submitted a low-rated review, returned the product, but have not yet received a refund." # Q11
    ]