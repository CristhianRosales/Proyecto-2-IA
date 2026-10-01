# Script para subir el proyecto a GitHub
# Ejecutar en una nueva ventana de PowerShell desde la carpeta ship_detection

Set-Location "c:\Users\USER\Desktop\Inteligencia Artificial\1\ship_detection"

git init
git remote add origin https://github.com/CristhianRosales/Proyecto-2-IA.git
git add .
git commit -m "Initial commit: AeroShip AI - Clasificador binario de barcos en imagenes satelitales"
git branch -M main
git push -u origin main

Write-Host "Subido exitosamente a GitHub!" -ForegroundColor Green
Write-Host "Enlace: https://github.com/CristhianRosales/Proyecto-2-IA" -ForegroundColor Cyan
