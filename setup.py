import os

# Actualizamos el instalador de paquetes
print("[+] Updating pip...")
os.system("python.exe -m pip install --upgrade pip")
print("[+] Done")

# Instalamos las dependencias necesarias
print("[+] Installing all the dependencies...")
os.system("pip install -r requirements.txt")
print("[+] Done")
