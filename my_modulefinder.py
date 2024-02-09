from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script('path_to_your_script.py')

print('Loaded modules:')
for name, mod in finder.modules.items():
    print('%s: ' % name, mod.__file__)

print('\nMissing modules:')
print(finder.badmodules)
