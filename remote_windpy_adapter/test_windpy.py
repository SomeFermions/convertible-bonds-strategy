from WindPy import w

start_result = w.start()
print("start ErrorCode:", start_result.ErrorCode)
print("connected:", w.isconnected())

r = w.wsd("127061.SZ", "close,turn", "2026-07-01", "2026-07-03", "")
print("ErrorCode:", r.ErrorCode)
print("Codes:", r.Codes)
print("Fields:", r.Fields)
print("Times:", r.Times)
print("Data:", r.Data)