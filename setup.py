from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='Questgen',
      version='1.0.0',
      description='Question generator from any text',
      packages=['Questgen', 'Questgen.encoding', 'Questgen.mcq'],
      url="https://github.com/mr-CLK/QA-Generator.git",
      install_requires=required,
      package_data={'Questgen': ['questgen.py', 'mcq.py', 'train_gpu.py', 'encoding.py']}
      )
