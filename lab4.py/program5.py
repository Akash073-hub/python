kal=int(input("enter the numbers"))
for i in range(1,kal+1):
    print(" " * (kal - i)+ " " .join(str(num) for num in range (1,i+1)))
