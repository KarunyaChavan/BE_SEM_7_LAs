// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title EmployeeRegistry
 * @dev A robust, object-oriented contract to manage employee data using structs, arrays, and events.
 *      Demonstrates encapsulation, validation, and modular function design following OOP principles.
 */
contract EmployeeRegistry {

    // ---------------------------------------------------------------------
    // STRUCTS & STORAGE
    // ---------------------------------------------------------------------

    struct Employee {
        uint256 empId;
        string name;
        string position;
        uint256 salary;
    }

    // Dynamic array to store all employee records
    Employee[] private employeeList;

    // Mapping to track existence of an employee ID
    mapping(uint256 => bool) private empExists;

    // Mapping to store index of employee in array for direct access
    mapping(uint256 => uint256) private empIndex;

    // ---------------------------------------------------------------------
    // EVENTS
    // ---------------------------------------------------------------------

    event EmployeeAdded(uint256 empId, string name, string position, uint256 salary);
    event SalaryUpdated(uint256 empId, uint256 oldSalary, uint256 newSalary);
    event EmployeeRemoved(uint256 empId);

    // ---------------------------------------------------------------------
    // CORE FUNCTIONS
    // ---------------------------------------------------------------------

    /**
     * @notice Add a new employee to the registry.
     * @param _empId Unique employee ID.
     * @param _name Employee name.
     * @param _position Job position or title.
     * @param _salary Initial salary amount.
     */
    function addEmployee(
        uint256 _empId,
        string memory _name,
        string memory _position,
        uint256 _salary
    ) public {
        require(!empExists[_empId], "Employee ID already exists.");
        require(_empId > 0, "Invalid employee ID.");
        require(_salary > 0, "Salary must be greater than zero.");

        employeeList.push(Employee(_empId, _name, _position, _salary));
        empExists[_empId] = true;
        empIndex[_empId] = employeeList.length - 1;

        emit EmployeeAdded(_empId, _name, _position, _salary);
    }

    /**
     * @notice Update the salary of an existing employee.
     * @param _empId Employee ID to update.
     * @param _newSalary New salary value.
     */
    function updateSalary(uint256 _empId, uint256 _newSalary) public {
        require(empExists[_empId], "Employee not found.");
        require(_newSalary > 0, "Salary must be positive.");

        uint256 index = empIndex[_empId];
        uint256 oldSalary = employeeList[index].salary;
        employeeList[index].salary = _newSalary;

        emit SalaryUpdated(_empId, oldSalary, _newSalary);
    }

    /**
     * @notice Remove an employee record from the registry.
     * @param _empId Employee ID to remove.
     * @dev Uses swap-and-pop method to keep array consistent.
     */
    function removeEmployee(uint256 _empId) public {
        require(empExists[_empId], "Employee not found.");

        uint256 index = empIndex[_empId];
        uint256 lastIndex = employeeList.length - 1;

        // Swap and remove to maintain contiguous array
        if (index != lastIndex) {
            Employee memory lastEmp = employeeList[lastIndex];
            employeeList[index] = lastEmp;
            empIndex[lastEmp.empId] = index;
        }

        employeeList.pop();
        delete empExists[_empId];
        delete empIndex[_empId];

        emit EmployeeRemoved(_empId);
    }

    // ---------------------------------------------------------------------
    // VIEW FUNCTIONS
    // ---------------------------------------------------------------------

    
    //* @notice Retrieve details of a specific employee.
    //* @param _empId Employee ID to fetch.
    //* @return empId, name, position, salary - all details of the employee.
    
    function getEmployee(uint256 _empId)
        public
        view
        returns (uint256 empId, string memory name, string memory position, uint256 salary)
    {
        require(empExists[_empId], "Employee not found.");
        Employee memory emp = employeeList[empIndex[_empId]];
        return (emp.empId, emp.name, emp.position, emp.salary);
    }

    /**
     * @notice Retrieve all employees in readable parallel arrays.
     * @return ids Array of employee IDs.
     * @return names Array of employee names.
     * @return positions Array of positions.
     * @return salaries Array of salaries.
     */
    function getAllEmployeesReadable()
        public
        view
        returns (
            uint256[] memory ids,
            string[] memory names,
            string[] memory positions,
            uint256[] memory salaries
        )
    {
        uint256 count = employeeList.length;
        ids = new uint256[](count);
        names = new string[](count);
        positions = new string[](count);
        salaries = new uint256[](count);

        for (uint256 i = 0; i < count; i++) {
            Employee memory emp = employeeList[i];
            ids[i] = emp.empId;
            names[i] = emp.name;
            positions[i] = emp.position;
            salaries[i] = emp.salary;
        }
    }

    /**
     * @notice Returns the total number of employees.
     * @return count Total employee count.
     */
    function getEmployeeCount() public view returns (uint256 count) {
        return employeeList.length;
    }

    // ---------------------------------------------------------------------
    // FALLBACK & RECEIVE
    // ---------------------------------------------------------------------

    fallback() external payable {}
    receive() external payable {}
}
