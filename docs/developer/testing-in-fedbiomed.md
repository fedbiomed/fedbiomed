# Testing in Fed-BioMed

## Unit Tests

Unit tests are designed to test smaller code units that make up larger solutions. These tests focus on isolated parts of the system and should not depend on external components such as third-party libraries or database connections. Meaning that third party library shouldn't cause failure in tests or shouldn't be tested.  

When writing unit tests, it is important to consider the following aspects:

- Reliance on file system operations.
- Mocking remote API calls or database connections.
- Minimizing dependence on third-party libraries.
- Keeping the tests isolated from the overall system.
- Focusing on testing smaller unit functions within the code.

In unit-testing the focus is to find the bug that can be found at small software pieces in isolation. There is no definitive answer as to whether the code should be completely isolated from the system in unit tests. In some cases, it may be beneficial to mock objects such as databases, depending on the complexity and ease of mocking.

Isolation from system means: Isolate from 

- the database 
- third party dependencies
- In complex cases: modules/classes within the library 
- file system
- API calls


It is important that test cases should be simple and focus on testing a single method/function. Unit tests should not test the integration of different modules.

Please find the following guidelines to pay attention to while writing unit tests:

1. Mock file system operations.
2. Tests should not depend on the system.
3. Tests should focus on smaller units and only validate the result of the function being tested.
4. Test the result of the method/function without getting lost in implementation details.
5. Make sure abstract classes are tested separately by mocking abstract methods.
6. Test the methods that inherit from base/abstract classes, but not the ones that are already tested in the base/abstract class.
7. Pay attention to testing the functionality of the wrapper class instead of testing the integration of wrapped classes.
8. Try to use Fake classes as little as possible. Follow the guide in section 8 to decide if a Fake class is required.
9. Mock complex classes to reduce complexity.
10. Do not mock the methods of the class being tested unless you have to.
11. Instead of mocking lower layer API mock the higher-level instance that is used by the class being tested.
12. Write separate test cases for private methods if the method is complex and complicated to test through public methods
13. Focus on what the function does while testing methods that receive generic data types as input.
14. For methods that can take different data types as arguments, do not write additional tests with different data types if the function doesn't explicitly perform specific actions for those types.
15. If your tests are difficult to write or complex, rethink the design of the class you're testing, and ensure that your implementation respects the SOLID principles.
16. Please add a new unit tests if a new bug is discovered. 

### 1. File system operations
- Tests should not fail due to file system errors. Therefore, use mocking for write/read operations.
- Tests should trust that the module used for write/read operations has been tested and works correctly, as long as correct input values are provided. Therefore, tests should only verify that `write` or `read` methods are called with the correct arguments.

### 2. System-dependent tests
- Avoid testing data that depends on the system, such as process time, file size, or permissions.
- System-dependent tests also include testing GPU or CPU functionality.
    - Avoid writing tests that compare the number of processes or CPU count.
    - If there are functions that require testing GPU functionality, implement those tests as integration tests.

### 3. Focus on smaller units
- Each test function should test a single method of a class, function, or module.
- Test functions can execute other methods of the class being tested or other functions. However, the purpose should be to prepare the test state rather than testing those executions.
- If the method being tested executes private members of the class, please follow the section [Private methods of a class](#12-private-methods-of-a-class).

### 4. Focus on intention more than implementation
- Focus on the expected execution outcome of the method being tested.
- Validate the result of the function/method rather than the underlying implementation details.
- The purpose of the test should be to verify if the outcome is as expected given the inputs.

### 5. Testing Mix-in Base/Abstract Class
- Test abstract/base classes by mocking abstract methods:
```python
self.abstract_methods_patcher = patch.multiple(AbstractClass, __abstractmethods__=set())
```
- Test the remaining methods.
- If there are remaining private methods, please test them explicitly rather than testing through the class that implements the abstract/base class. This avoids duplicating tests among the classes that inherit from the abstract/base class being tested.

### 6. Testing classes that implement Base/Abstract classes
- Following item 5, avoid testing the methods that are already tested in the TestCase of the abstract/base class.
- Test the methods that are overwritten by the class being tested.
- When testing extended methods, please follow the principles in [Item 3 - Focus on smaller units](#3-focus-on-smaller-units).

### 7. Testing Wrappers

#### 7.1 Wrappers that wrap objects with a COMMON interface
- Focus on wrapper methods rather than the wrapped objects. As objects that can be wrapped by the wrapper have a common interface, use a single object instead of applying the same test for each of them.
- If the wrapped object is complex and requires different states, consider creating a fake object, and follow the section [Using Fake objects/classes](#8-using-fake-objects-classes) to determine if a fake object is necessary.

#### 7.2 Wrappers that wrap objects with DIFFERENT interfaces
- If the wrapper wraps different objects with different interfaces (methods), you can write tests for each wrapped object.
- Follow the same logic as in "7.1" for mocking or using fake objects.

### 8. Using Fake or Stub objects/classes
- Fake or Stub objects require coherency between the real implementation and the fake one. Therefore, avoid using fake objects as much as possible because they can be difficult to maintain.
- Please answer the following questions before deciding to create a fake class:
    - 1. Is the object passed to a function or class as an argument?
    - 2. Is the object dependent on the system or external actions (e.g., database)?
    - 3. Does the object have too many dependencies on other modules of the library?
- If 1. and 2. or 3. are "yes," feel free to create a fake object.
- If 1. is "no" and the others are "yes," consider using mocking instead of fake or stub objects.

### 9. Using Mocked objects
- Mocking helps isolate the test from the environment and easily test different scenarios of a single function.
- Pay attention to using the `spec`, `spec_set`, or `auto_spec` functionalities of `unittest.mock.Mock`. These functionalities help mock objects that respect the specifications of the real class, so the mock will be updated if an API of the real class changes.
- Using too many mock objects may not be a good idea because:
    - It may increase the complexity of the test function.
    - It may isolate the test too much from the remaining parts of the software, resulting in invalid tests.

- Before deciding to mock an object, please answer the following questions:
    - 1. Is the object instantiated or used directly within the module or class being tested?
    - 2. Is the object dependent on the system or external actions (e.g., database, network etc)?
    - 3. Does the object have too many dependencies on other modules of the library?
    - 4. Does the method being tested require many results or states of the object that needs to be mocked?
    - 5. Is mocking saving you a lot of time and reducing complexity?

- If 1. is no and any other condition is yes, consider using a fake class.
- If 1. is yes and any other condition is yes, consider using mocking. 
- If mocking changes the intention of the test, try to find another solution besides mocking or using fake class.

### 10. Avoid mocking the methods of the class being tested
- Please avoid mocking public/private methods of the class being tested.
- You can mock in some cases if you think that you have to mock it to reduce complexity. However, the need for mocking can be due to some design issues. Therefore, please make sure that the class being tested, or the test case is designed well. 

### 11. Leveling Mocks
- The classes being tested may require instantiating other instances that contain methods that need to be mocked, such as a database connection or file read/write operation. In such scenarios, instead of mocking the built-in read/write methods, mock the higher-level instance that is used by the class being tested.

### 12. Private methods of a class
- Write separate tests for private methods if it has complicated state preparation.
- For some cases, if a private method is not complex and easy to cover every line of code, you can test them through public method. However, there is no need to test for the second time through another public method if it is already tested once. 

### 13. Generic Classes that Accept Different Data Types (e.g., Dict)
- When testing generic classes that can accept different data types, such as dictionaries containing different types of data, consider the following:
    - Test different data types if the behavior of the class depends on the specific data types.
    - If the class uses another class to manage different data types, and that class has been tested, there is no need to test all possible combinations of data types.

### 14. Testing Different Methods that Accept Different Data Types for a Single Argument
- Test each data type separately (e.g., float, int) to cover all possible scenarios.
- IMPORTANT: If the behavior of the method is consistent across different data types, there is no need to test all combinations.

### 15. Having complex unit test cases in the end
- If unit tests require too much mocking and complexity, it is better to reconsider the design chosen for the class being tested. Following SOLID principles while designing a class or a module can help.
    - Single Responsibility Principle
        - A class should have only one reason to change.
    - Open-Closed Principle
        - Classes should be open for extension and closed to modification. Adding new functionality should not require modifying the existing code for the class, as it introduces potential bugs.
    - Liskov Substitution Principle
        - Subclasses should be substitutable for their base classes.
    - Interface Segregation Principle
        - Specific interfaces are better than one general-purpose interface.
    - Dependency Inversion Principle
        - Classes should depend on interfaces or abstract classes instead of concrete classes and functions.

### 16. Learn from bugs 
- When a bug is discovered please consider adding a unit tests for it. However, if the bug can not be tested using unit tests please consider adding integration or end-to-end test. 

## Integration Testing

Integration testing involves testing the interaction between software modules. In the context of Fed-BioMed, integration testing can be performed on various integrated modules such as nodes, rounds, data loaders, training plans, models, optimizers, command-line interfaces (CLIs), dataset managers, model registration/approval components, and certificate registration components. The selection of integrated modules for testing depends on the logical integration of the system. 


- Use database
- Write and read files from file system
- Rely on 3rd party libraries 

The environment and the API for integration tests are still in working process.

## End-to-End Testing

End-to-end testing focuses on testing the entire system or application as a whole, ensuring that all components work together correctly. In the context of Fed-BioMed, end-to-end testing would involve testing the complete workflow and functionalities of the federated learning framework, including data collection, model training, aggregation, and evaluation.
