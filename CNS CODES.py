#1
s="Crptography"
ans=""
n=len(s)
for i in range(n):
    ans+=chr(ord(s[i])^0)
print(s)
print(ans)

#2
s="Cryptography and N S" 
ans=""
ans1=""
ans2=""
n=len(s)
for i in range(n):
    ans+=chr(ord(s[i])^127)
for i in range(n):
    ans1+=chr(ord(s[i])|127)
for i in range(n):
    ans2+=chr(ord(s[i])&127)
print(s)
print(ans)
print(ans1)
print(ans2)

#3 caeser
def encrypt(m,s):
    res = ""
    for item in m:
        if item.isupper():
            res = res + chr((ord(item)+s-65)%26 +65)
        else:
            res = res + chr((ord(item)+s-97)%26 +97)
    return res
def decrypt(ciphertext, s):
    res = ""
    for item in ciphertext:
        if item.isupper():
            res = res + chr((ord(item) - s - 65) % 26 + 65)
        else:
            res = res + chr((ord(item) - s - 97) % 26 + 97)
    return res
m=input("Enter plaintext: ")
s=int(input("Enter shift: "))
e=encrypt(m,s)
print("Encryption: ",e)
d=decrypt(e,s)
print("Decryption: ",d)

#4 substitution
def encrypt (plain,key):
    cipher=""
    for i in range(len(plain)):
        if plain[i]==' ':
            cipher+=' '
            continue
        cipher+=chr((ord(plain[i])+ord(key[i])-2*97)%26+97)
    return cipher 
def decrypt (cipher,key):
    plain=''
    for i in range(len(cipher)):
        if cipher [i]==' ':
            plain+=' '
            continue
        plain+=chr((ord(cipher[i])-ord(key[i]))%26+97)
    return plain
plaintext=input ()
key=input ()
cipher=encrypt (plaintext.lower (),key)
print('Encrypted text:' ,cipher)
print ('Decrypted text:', decrypt (cipher.lower (), key))

#5 hill
import numpy
def create_matrix_from(key):
    m=[[0] * 3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            m[i][j] = ord(key[3*i+j]) % 65
    return m
# C = P*K mod 26
def encrypt(P, K):
    C=[0,0,0]
    C[0] = (K[0][0]*P[0] + K[1][0]*P[1] + K[2][0]*P[2]) % 26
    C[1] = (K[0][1]*P[0] + K[1][1]*P[1] + K[2][1]*P[2]) % 26
    C[2] = (K[0][2]*P[0] + K[1][2]*P[1] + K[2][2]*P[2]) % 26
    return C

def Hill(message, K):
    cipher_text = []
    
    for i in range(0,len(message), 3):
        P=[0, 0, 0]
        
        for j in range(3):
            P[j] = ord(message[i+j]) % 65
        
        C = encrypt(P,K)
        
        for j in range(3):
            cipher_text.append(chr(C[j] + 65))

    return "".join(cipher_text)

def MatrixInverse(K):
    det = int(numpy.linalg.det(K))
    det_multiplicative_inverse = pow(det, -1, 26)
    K_inv = [[0] * 3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            Dji = K
            Dji = numpy.delete(Dji, (j), axis=0)
            Dji = numpy.delete(Dji, (i), axis=1)
            det = Dji[0][0]*Dji[1][1] - Dji[0][1]*Dji[1][0]
            K_inv[i][j] = (det_multiplicative_inverse * pow(-1,i+j) * det) % 26
    return K_inv


message = "MYSECRETMESSAGE"
key = "RRFCCTVSV"
#Create the matrix K that will be used as the key
K = create_matrix_from(key)
print(K)
# C = P * K mod 26
cipher_text = Hill(message, K)
print ('Cipher text: ', cipher_text)
# Decrypt
# P = C * K^-1 mod 26
K_inv = MatrixInverse(K)            
plain_text = Hill(cipher_text, K_inv)
print ('Plain text: ', plain_text)
M = (numpy.dot(K,K_inv))
for i in range(3):
    for j in range(3):
        M[i][j] = M[i][j] % 26
print(M)

#6 playfair
def playfair_cipher(plaintext, key, mode):  

    alphabet = 'abcdefghiklmnopqrstuvwxyz'  
 
    key = key.lower().replace(' ', '').replace('j', 'i')  
 
    key_square = ''  
    for letter in key + alphabet:  
        if letter not in key_square:  
            key_square += letter  
 
    plaintext = plaintext.lower().replace(' ', '').replace('j', 'i')
    
    if len(plaintext) % 2 == 1:  
        plaintext += 'x'  
    digraphs = [plaintext[i:i+2] for i in range(0, len(plaintext), 2)]  

    def encrypt(digraph):  
        a, b = digraph  
        row_a, col_a = divmod(key_square.index(a), 5)  
        row_b, col_b = divmod(key_square.index(b), 5)  
        if row_a == row_b:  
            col_a = (col_a + 1) % 5  
            col_b = (col_b + 1) % 5  
        elif col_a == col_b:  
            row_a = (row_a + 1) % 5  
            row_b = (row_b + 1) % 5  
        else:  
            col_a, col_b = col_b, col_a  
        return key_square[row_a*5+col_a] + key_square[row_b*5+col_b]
    
    def decrypt(digraph):  
        a, b = digraph  
        row_a, col_a = divmod(key_square.index(a), 5)  
        row_b, col_b = divmod(key_square.index(b), 5)  
        if row_a == row_b:  
            col_a = (col_a - 1) % 5  
            col_b = (col_b - 1) % 5  
        elif col_a == col_b:  
            row_a = (row_a - 1) % 5  
            row_b = (row_b - 1) % 5  
        else:  
            col_a, col_b = col_b, col_a  
        return key_square[row_a*5+col_a] + key_square[row_b*5+col_b]  
 
    result = ''  
    for digraph in digraphs:  
        if mode == 'encrypt':  
            result += encrypt(digraph)  
        elif mode == 'decrypt':  
            result += decrypt(digraph)  
                                          
    return result  
# Example usage  
plaintext = 'She sells sea shells at the sea shore'  
key = 'example key'  
ciphertext = playfair_cipher(plaintext, key, 'encrypt')  
print(ciphertext) # outputs: "iisggymlgmsyjqu"  
decrypted_text = playfair_cipher(ciphertext, key, 'decrypt')  
print(decrypted_text)# (Note: 'x' is added as padding)

#7 des
from Crypto.Cipher import DES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad,unpad

def en(pt,key):
    cipher=DES.new(key,DES.MODE_ECB)
    padded_text=pad(pt,DES.block_size)
    encrypted=cipher.encrypt(padded_text)
    return encrypted

def de(ci,key):
    decipher=DES.new(key,DES.MODE_ECB)
    unpadded=decipher.decrypt(ci)
    decrypted=unpad(unpadded,DES.block_size)
    return decrypted

key=get_random_bytes(8)
pt=b"Ee Sala Cup Namde"
ci=en(pt,key)
print(ci)
print(de(ci,key).decode('utf-8'))

#8 aes
import os
import pyaes
# Generate a random 128-bit (16-byte) AES key
key = os.urandom(16)
# Create an AES cipher object
aes = pyaes.AESModeOfOperationCTR(key)
# Plaintext to encrypt
plaintext = "hello world"
# Encrypt the plaintext
cipherText = aes.encrypt(plaintext.encode('utf-8'))  # Encode plaintext as bytes
# Decrypt the ciphertext
decrypted = aes.decrypt(cipherText).decode('utf-8')  # Decode the decrypted bytes to string
print("Original plaintext:", plaintext)
print("Encrypted ciphertext:", repr(cipherText))
print("Decrypted plaintext:", decrypted)

#9 blowfish
from Crypto.Cipher import Blowfish
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encrypt_text(key, plaintext):
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    padded_plaintext = pad(plaintext, Blowfish.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext

def decrypt_text(key, ciphertext):
    decipher = Blowfish.new(key, Blowfish.MODE_ECB)
    decrypted_text = decipher.decrypt(ciphertext)
    original_plaintext = unpad(decrypted_text, Blowfish.block_size)
    return original_plaintext

# Generate a random 64-bit (8-byte) Blowfish key
key = get_random_bytes(8)
# Define the new plaintext to be encrypted
new_plaintext = b"This is a different plaintext."
# Encrypt the plaintext
encrypted_data = encrypt_text(key, new_plaintext)
# Decrypt the ciphertext
decrypted_data = decrypt_text(key, encrypted_data)
# Print the results
print("Original plaintext:", new_plaintext)
print("Encrypted ciphertext:", encrypted_data)
print("Decrypted text:", decrypted_data.decode("utf-8"))

#10 rc4
def display(disp):
    convert = [chr(char) for char in disp]
    print("".join(convert))

def main():
    temp = 0
    ptext = input("\nEnter Plain Text: ")
    key = input("Enter Key Text: ")
    ptextLen = len(ptext)
    keyLen = len(key)
    cipher = [0] * ptextLen
    decrypt = [0] * ptextLen
    ptexti = [ord(char) for char in ptext]
    keyi = [ord(char) for char in key]
    s = list(range(256))
    k = keyi * (255 // keyLen + 1)
    j = 0

    for i in range(256):
        j = (j + s[i] + k[i]) % 256
        s[i],s[j]=s[j],s[i]

    i = 0
    j = 0
    z = 0

    for l in range(ptextLen):
        i = (l + 1) % 256
        j = (j + s[i]) % 256
        temp = s[i]
        s[i] = s[j]
        s[j] = temp
        z = s[(s[i] + s[j]) % 256]
        cipher[l] = z ^ ptexti[l]
        decrypt[l] = z ^ cipher[l]

    print("ENCRYPTED: ", end="")
    display(cipher)
    print("\nDECRYPTED: ", end="")
    display(decrypt)

if __name__ == "__main__":
    main()
