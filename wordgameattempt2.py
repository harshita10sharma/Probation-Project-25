import random
guess = random.choice(["python", "machine", "learning", "data", "science"])
print(guess)
n= len(guess)
name = ""  
chance = 6
i = 0      
while chance > 0 and i < n:
    user = input("Enter the guessed letter ")

    if user == guess[i]:   
        name += user
        i += 1
        print("Correct so far:", name)
        if name == guess:
            print("Congratulations! You guessed the correct word:", guess)
            break
        else:
         print("Try again")
    else:
        chance -= 1
        print("Wrong guess! Chances left:", chance)

if name != guess:
    print("Game over! The word was:", guess)
