-- params: ["%Quý 2/2026%"]
SELECT COUNT(DISTINCT "Status") AS metric FROM "danh_sach_hoa_don" WHERE LOWER(CAST("Status" AS VARCHAR)) LIKE LOWER(?)
