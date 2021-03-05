rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''

map =[rock, paper , scissors]
import random
user_input  = int(input("Choose your option 0 for rock,2 for scissors and 1 for paper\n "))

if user_input >=3 or user_input<0:
  print("invalid")
else:
  print( map[user_input])

computer_input = random.randint(0,2)
print(f"Computer chose \n{computer_input}")

print(map[computer_input])

if user_input == 0 and computer_input ==2:
  print("You win")
elif computer_input> user_input:
  print("You lose")
elif computer_input == 0 and user_input == 1:
  print("You lose")
elif computer_input ==  user_input:
  print("Draw")
