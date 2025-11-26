import json
from typing import Any, Dict, Optional
from insight_extraction.prompts.extraction_prompt import build_extraction_prompt


class SQLQueryGenerator:
    """
    Wrapper di alto livello che usa OpenAILLMClient e build_sql_prompt
    per ottenere il codice SQL dal modello.
    """

    def __init__(
        self,
        llm_client: Any,
        sql_dialect: str = "SQLite",
    ) -> None:

        self.llm_client = llm_client or OpenAILLMClient()
        self.sql_dialect = sql_dialect

    def generate_sql(
        self,
        user_question: str,
        json_spec: Dict[str, Any],

        main_table: str,
        schema_text: Optional[Any] = None,
    ) -> str:
        """
        Costruisce il prompt completo, chiama il modello e restituisce
        il codice SQL (potenzialmente più query).
        """
        prompt = build_extraction_prompt(
            user_question=user_question,
            json_spec=json_spec,
     
            main_table=main_table,
            sql_dialect=self.sql_dialect,
            schema_text=schema_text,
        )

        sql_code = self.llm_client.invoke(prompt)
        return sql_code