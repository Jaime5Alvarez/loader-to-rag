import polars as pl
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


async def load_excel_async(file_path: str):

    sheets_dict = await asyncio.to_thread(pl.read_excel, file_path, sheet_id=0)
    return sheets_dict


async def process_excel_to_documents(file_path: str):
    sheets_dict = await load_excel_async(file_path)

    data = []
    for sheet_name, df in sheets_dict.items():
        records = df.to_dicts()
        
        for idx, row_dict in enumerate(records):
            content = "\n".join([f"{col}: {val}" for col, val in row_dict.items()])
            
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "sheet": sheet_name, "row": idx},
            )
            data.append(doc)

    return data


async def process_and_split_excel(
    file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
):
    data = await process_excel_to_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    splits = text_splitter.split_documents(data)

    return splits


async def main():
    splits = await process_and_split_excel("xlsx_loader/FSI-2023-DOWNLOAD.xlsx")

    for split in splits:
        sheet = split.metadata.get("sheet", "N/A")
        row = split.metadata.get("row", "N/A")
        content = split.page_content
        print(f"Sheet '{sheet}', Row {row}: {content}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
