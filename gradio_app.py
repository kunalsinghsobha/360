import gradio as gr
import pandas as pd
from customer360.units import build_units
from customer360.resolution import resolve_customers
from customer360.scoring import compute_customer_rollups, ai_like_summary

def process(file):
    path = file.name
    df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path, encoding="utf-8-sig")
    df_units, units_master, unit_bookings_map = build_units(df)
    df["customer_master_id"] = resolve_customers(df)
    rollups = compute_customer_rollups(df_units, units_master)
    try:
        summary = "\n".join([ai_like_summary(r) for _, r in rollups.head(20).iterrows()])
    except Exception as e:
        summary = f"Summary failed: {e}"
    return df_units.head(50), units_master.head(50), rollups.head(50), summary

with gr.Blocks(title="Customer 360") as demo:
    gr.Markdown("# Customer 360 — Quick Test")
    inp = gr.File(label="Upload Salesforce report (.csv/.xlsx)")
    btn = gr.Button("Run")
    out1 = gr.Dataframe(label="df_units (head)")
    out2 = gr.Dataframe(label="units_master (head)")
    out3 = gr.Dataframe(label="customer_rollups (head)")
    out4 = gr.Textbox(label="Summary", lines=10)
    btn.click(process, [inp], [out1, out2, out3, out4])

if __name__ == "__main__":
    demo.launch()
