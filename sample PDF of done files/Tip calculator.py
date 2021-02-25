print ("Welcome to the tip calculator")
Bill = int(input("what is the bill?\n"))
ppl = int(input("How many ppl pay?"))
Tip = int(input("Tell me the tip? \n"))
tip2 = float(Tip/100)
pay = (Bill/ppl)
total_tip = 1+tip2
You_pay = round(pay * total_tip)
print(f"Each pay $ {You_pay}")