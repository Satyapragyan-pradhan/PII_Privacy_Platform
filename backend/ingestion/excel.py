import io

import pandas as pd


def extract_excel_text(content: bytes):
    excel_file = io.BytesIO(content)

    sheets = pd.read_excel(
        excel_file,
        sheet_name=None,
        header=None
    )

    text_parts = []

    for sheet_name, dataframe in sheets.items():

        text_parts.append(
            f"Sheet: {sheet_name}"
        )

        for row in dataframe.itertuples(
            index=False,
            name=None
        ):
            values = []

            for value in row:
                if pd.notna(value):
                    values.append(str(value))

            if values:
                text_parts.append(
                    " | ".join(values)
                )

    return {
        "text": "\n".join(text_parts),
        "needs_ocr": False,
        "pages": []
    }