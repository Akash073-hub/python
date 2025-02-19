kal=int(input("enter the number"))
for j in range(kal-1,0,-1):
    print(" " * (kal - j)+ "*" * (2*j-1))
