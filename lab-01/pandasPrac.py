#CS21B2019 DSLR SIDDESH
import numpy as np
import pandas as pd

data = {
    'Name': ['Ram', 'Sam', 'Prabhu'],
    'Account_Number': [9893893891, 9893893898, 9893893871],
    'Account_type': ['SB', 'CA', 'SB'],
    'Adhaar_No': [9593893891, 9593893891, 9593893891],
    'Balance': [8989839, 7690990, 989330]
}

df = pd.DataFrame(data)
df.to_csv('SBIAccountHolder.csv', index = False)

def CheckAccountExists(acc_no):
    if df[df.Account_Number == acc_no].empty:
        return False
    return True

def AppendRecord():
    name = input("Enter name : ")
    acc_no = int(input("Enter account number : "))
    if CheckAccountExists(acc_no):
        print("Account already exists")
        return
    acc_type = input("Enter account type SB/CA: ")
    adhaar_no = int(input("Enter adhaar number : "))
    if not df[df.Adhaar_No == adhaar_no].empty:
        print("Adhaar number already exists")
        return
    balance = int(input("Enter balance : "))
    if balance < 0:
        print("Invalid balance!!")
        return

    df.loc[len(df.index)] = [name, acc_no, acc_type, adhaar_no, balance]
    df.to_csv('SBIAccountHolder.csv', index=False)
    print("...Record appended successfully...")

def DeleteRecord():
    acc_no = int(input("Enter account number : "))
    if not CheckAccountExists(acc_no):
        print("Account does not exist!!")
        return
    df.drop(df[df.Account_Number == acc_no].index, inplace=True)
    df.to_csv('SBIAccountHolder.csv', index=False)
    df.reset_index(drop=True, inplace=True)
    print("...Record deleted successfully...")

def Credit():
    acc_no = int(input("Enter account number : "))
    if not CheckAccountExists(acc_no):
        print("Account does not exist!!")
        return
    amount = int(input("Enter amount to be credited : "))
    if amount < 0:
        print("Invalid amount!!")
        return
    df.loc[df.Account_Number == acc_no, 'Balance'] += amount
    df.to_csv('SBIAccountHolder.csv', index=False)
    print("...Amount credited successfully...")

def Debit():
    acc_no = int(input("Enter account number : "))
    if not CheckAccountExists(acc_no):
        print("Account does not exist!!")
        return
    amount = int(input("Enter amount to be debited : "))
    if amount < 0:
        print("Invalid amount!!")
        return
    if df.loc[df['Account_Number'] == acc_no, 'Account_type'].iloc[0] == 'SB':
        if df.loc[df['Account_Number'] == acc_no, 'Balance'].iloc[0] - amount < 0:
            print("Insufficient balance!!")
            return
    df.loc[df.Account_Number == acc_no, 'Balance'] -= amount
    df.to_csv('SBIAccountHolder.csv', index=False)
    print("...Amount debited...")
    
def GetAccountDetails():
    acc_no = int(input("Enter account number : "))
    if not CheckAccountExists(acc_no):
        print("Account does not exist!!")
        return
    print("\n",df[df.Account_Number == acc_no].to_string(index=False))

if __name__ == "__main__":

    print("1.Append Record")
    print("2.Delete Record")
    print("3.Credit")
    print("4.Debit")
    print("5.Get Account Details")
    print("6.Exit")

    while True:
        choice = int(input("\nEnter your choice : "))
        if choice == 1:
            AppendRecord()
        elif choice == 2:
            DeleteRecord()
        elif choice == 3:
            Credit()
        elif choice == 4:
            Debit()
        elif choice == 5:
            GetAccountDetails()
        elif choice == 6:
            print("Exiting..")
            break
        else:
            print("Invalid choice!!")