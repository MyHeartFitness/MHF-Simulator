from io import BytesIO
import pandas as pd
from datetime import datetime
import json

def to_excel(profiles: pd.DataFrame, params: dict) -> bytes:
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n": len(profiles),
        "parameters": params,
    }
    summary = profiles.describe(include="all").T

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as xw:
        profiles.to_excel(xw, index=False, sheet_name="Profiles")
        summary.to_excel(xw, sheet_name="Summary")
        pd.DataFrame({"metadata_json":[json.dumps(meta, indent=2)]}).to_excel(
            xw, index=False, sheet_name="Metadata"
        )
    return buffer.getvalue()
