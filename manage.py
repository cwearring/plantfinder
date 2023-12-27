import argparse

def deploy(runinit = False):
	"""Run deployment tasks."""
	from app import create_app,db
	from flask_migrate import upgrade,migrate,init,stamp
	from app.models import User, SessionData

	app = create_app()
	app.app_context().push()
	db.create_all()

	# init database first time
	if runinit:
		init()
		print("shit")
	# migrate database to latest revision
	stamp()
	migrate()
	upgrade()
	
def main():
	# Create an ArgumentParser object
	parser = argparse.ArgumentParser(description='Run deployment tasks.')

	# Add a command-line argument for the deploy function
	parser.add_argument('--dbinit', action='store_true', help='Run deployment tasks.')

	# Parse the command-line arguments
	args = parser.parse_args()
	print("shit1")
	# Check if the --deploy flag is provided
	if args.dbinit:
		deploy(True)
	else:
		deploy(False)

if __name__ == '__main__':
	main()
	