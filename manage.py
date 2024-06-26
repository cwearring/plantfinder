import argparse
import os
from dotenv import load_dotenv

def deploy(dbinit = False):
	"""Run deployment tasks."""
	from app import create_app, db
	from flask_migrate import upgrade, migrate, init, stamp

	# get settings from the .env file 
	load_dotenv()
	my_config = os.getenv('CONFIG_LEVEL', None)
	print(f"my_config in deploy = {my_config} - deploy")

	app = create_app(config_name = my_config)
	app.app_context().push()
	db.create_all()

	# init database first time 	print(f"init = {dbinit}")
	if dbinit:
		init()
	# migrate database to latest revision
	stamp()
	migrate()
	upgrade()
	
def main():
	# Create an ArgumentParser object
	parser = argparse.ArgumentParser(description='Run deployment tasks.')

	# Add a command-line argument for the deploy function
	parser.add_argument('--dbinit', action='store_true', help='Run dB deployment tasks.')

	# Parse the command-line arguments
	args = parser.parse_args()
	print(args)

	# Check if the --deploy flag is provided
	if args.dbinit:
		deploy(True)
	else:
		deploy(False)

if __name__ == '__main__':
	main()
	