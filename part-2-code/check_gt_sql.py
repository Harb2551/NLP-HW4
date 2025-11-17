"""
Removed: this debug helper was tied to the custom tokenizer workflow.
"""

raise SystemExit("check_gt_sql.py removed as part of reverting custom tokenizer changes.")
#!/usr/bin/env python3
"""
Check GT SQL preservation in the current pipeline
"""

import sys
sys.path.append('/Users/hb25/Desktop/HW_S3/NLP/HW4/hw4/hw4-code/part-2-code')

from transformers import T5TokenizerFast
import os

def check_gt_sql_preservation():
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
    else:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    
    # Example GT SQL
    gt_sql = "SELECT DISTINCT flight_1.flight_id FROM flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2, days days_1, date_day date_day_1 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'DENVER' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'PHILADELPHIA' AND flight_1.flight_days = days_1.days_code AND days_1.day_name = date_day_1.day_name AND date_day_1.year = 1991 AND date_day_1.month_number = 1 AND date_day_1.day_number = 20 ) END"
    
    # Simulate decode from model target
    tokens = tokenizer.encode(gt_sql, return_tensors="pt")
    decoded = tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
    # Remove END token as in eval_utils.py
    cleaned = decoded.replace(' END', '').strip()

    def restore_sql_spacing(sql):
        import re
        # Replace SELECT_DISTINCT with SELECT DISTINCT
        sql = sql.replace('SELECT_DISTINCT', 'SELECT DISTINCT')
        # Add space before and after SQL keywords
        keywords = [
            "SELECT", "DISTINCT", "FROM", "WHERE", "AND", "OR", "JOIN", "INNER JOIN", "LEFT JOIN", "ON",
            "GROUP BY", "ORDER BY", "HAVING", "COUNT", "SUM", "AVG", "MAX", "MIN"
        ]
        for kw in keywords:
            sql = re.sub(rf"(?<! )({kw})", r" \1", sql)
        # Add space after commas and parentheses
        sql = re.sub(r",", ", ", sql)
        sql = re.sub(r"\(", "( ", sql)
        sql = re.sub(r"\)", " )", sql)
        # Add space around operators
        sql = re.sub(r"([=<>!]+)", r" \1 ", sql)
        # Remove multiple spaces
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    fixed_sql = restore_sql_spacing(cleaned)

    print("Original GT SQL:")
    print(gt_sql)
    print("\nDecoded and cleaned GT SQL:")
    print(cleaned)
    print("\nPost-processed SQL:")
    print(fixed_sql)

    # Check for corruption
    if "FROM flight flight_1" in fixed_sql and "WHERE flight_1.from_airport" in fixed_sql and "SELECT DISTINCT" in fixed_sql:
        print("\n✅ GT SQL is preserved and fixed correctly!")
    else:
        print("\n❌ GT SQL is still corrupted!")

if __name__ == "__main__":
    check_gt_sql_preservation()