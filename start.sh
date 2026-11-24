docker run --name puls-events-api \
 -p 127.0.0.1:8001:8000 \
 --env-file .env.docker \
 #--mount "type=bind,"source=$nibaj@Oxlade MINGW64 ~/Openclassrooms/Projet9_systeme_rag,target=/app/data" \
 puls-events-rag 