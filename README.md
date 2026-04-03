if you dont ahve uv install it using this command curl -LsSf https://astral.sh/uv/install.sh | sh
first step - create a venv using uv venv
then install the dependencies uisng uv pip install -r requirements.txt
then use the getmodel.py to download your specific model uv run getmodel.py
then you do inference using uv run inference.py
