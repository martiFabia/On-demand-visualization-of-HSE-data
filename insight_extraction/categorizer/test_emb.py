from __future__ import annotations

import json
from pathlib import Path

from categorizer import run_pipeline




def main():
    print("🔍 Avvio test completo della pipeline…")

    excel_path = Path("data/test_observations.xlsx")  # <-- REPLACE with your path
    intent_data = load_test_intent()
    intent_tmp_path = Path("categorizer/tests/tmp_intent.json")

    # temporarily save the intent
    with intent_tmp_path.open("w", encoding="utf-8") as f:
        json.dump(intent_data, f, indent=2, ensure_ascii=False)

    output_path = Path("categorizer/tests/output_assignments.json")

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_tmp_path,
        output_path=output_path,
    
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.4,
        min_support_ratio=0.01,
        max_examples=None,  # or set a number
    )

    print(f"🎉 Test completato. Risultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
