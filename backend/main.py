
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def root():
    return ""

@app.get("/health")
def health_check():
    return {"status": "ok"}




# from pydantic import BaseModel

# app = FastAPI()

# class Item(BaseModel):
#     text: str = None
#     is_done: bool = False

# items = []

# @app.get("/")
# def root():
#     return {"Hello": "World"}


# @app.get("/items", response_model=list[Item])
# def list_items(limit: int = 10):
#     return items[0:limit]

# @app.post("/items")
# def create_item(item: Item):
#     items.append(item)
#     return items

# @app.get("/items/{item_id}", response_model = Item)
# def get_item(item_id: int) -> Item:
#     if item_id < len(items):
#         item = items[item_id]

#     else:
#         raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

#     return item