import pytest, os
from src.inference.execution_UI import CodeExecutor

@pytest.fixture(scope="module")
def executor():
    return CodeExecutor()

@pytest.fixture(scope="module")
def setup_environment(executor):
    executor.variables = {
        'test_int': {'value': 42},
        'test_str': {'value': "hello"}
    }
    executor.save_directory = './test_tmp'
    os.makedirs(executor.save_directory, exist_ok=True)
    return executor

class TestCodeExecutor:
    def test_save_environment(self, setup_environment):
        file_path = os.path.join(setup_environment.save_directory, "env.pkl")
        setup_environment.save_environment(file_path)
        assert os.path.exists(file_path), "Environment file was not created"

    def test_load_environment(self, setup_environment):
        file_path = os.path.join(setup_environment.save_directory, "env.pkl")
        setup_environment.save_environment(file_path)

        new_executor = CodeExecutor()
        new_executor.save_directory = './test_tmp'
        new_executor.load_environment(file_path)

        assert new_executor.variables == setup_environment.variables, "Variables do not match after loading"
        assert new_executor.counter == setup_environment.counter, "Counters do not match after loading"

    def test_nonexistent_load_environment(self, executor):
        with pytest.raises(FileNotFoundError):
            executor.load_environment("nonexistent.pkl")

    def test_filter_picklable_variables(self, executor):
        executor.variables['test_int'] = 42
        executor.variables['test_str'] = "hello"
        executor.variables['test_unpicklable'] = lambda x: x  
        
        picklable_vars = executor.filter_picklable_variables()
        assert 'test_int' in picklable_vars
        assert 'test_str' in picklable_vars
        assert 'test_unpicklable' not in picklable_vars, "Unpicklable variables should be filtered out"

@pytest.mark.usefixtures("setup_environment")
class TestExecution:
    def test_execute_api_call(self, executor):
        code = "a=1"
        result = executor.execute_api_call(code, "code")
        assert 'a' in list(executor.variables.keys()), "a variable should be imported into globals"