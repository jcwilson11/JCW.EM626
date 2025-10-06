name = input("Enter your name: ")
age = int(input("Enter your age: "))
if age >= 18:
    print(name + " is an adult")
else:
    print(name + " is a minor")

# Step 1: Identify the problems with this script from a softwareengineering perspective (e.g., no modularity, no separation of concerns)
'''
no error handling for non-integer age input
no functions to encapsulate logic
'''

# Step 2: restructure outline 

# Step 3: explain which SWE principles you applied  