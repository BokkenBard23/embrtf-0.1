-- Найти все диалоги, где клиент обещает перезвонить ЧЁТКО (родительский запрос)
WITH client_promises AS (
    SELECT DISTINCT d.id, d.text
    FROM dialogs d
    JOIN callback_phrases cp ON d.text LIKE '%' || cp.phrase || '%'
    WHERE cp.source = 'client_promise' AND cp.verified = 1
),
-- Исключить диалоги, где оператор предложил звонок или клиент дал неопределённость (дочерний запрос)
excluded_dialogs AS (
    SELECT DISTINCT d.id
    FROM dialogs d
    JOIN callback_phrases cp ON d.text LIKE '%' || cp.phrase || '%'
    WHERE cp.source IN ('client_uncertain', 'operator_offer') AND cp.verified = 1
)
-- Оставить только чистые ошибки
SELECT cp.id, cp.text
FROM client_promises cp
LEFT JOIN excluded_dialogs ed ON cp.id = ed.id
WHERE ed.id IS NULL;