import yaml, glob, sys

errors = []
for f in sorted(glob.glob('configs/*.yaml')):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            yaml.safe_load(fh)
        print('OK', f)
    except Exception as e:
        print('ERROR', f, e)
        errors.append((f, str(e)))

if errors:
    print('\nValidation failed for', len(errors), 'files')
    sys.exit(2)
print('\nAll configs valid')
sys.exit(0)
