mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = \$PORT\n\
[theme]\n\
primaryColor = '#3b82f6'\n\
backgroundColor = '#0f172a'\n\
secondaryBackgroundColor = '#1e293b'\n\
textColor = '#f1f5f9'\n\
" > ~/.streamlit/config.toml
