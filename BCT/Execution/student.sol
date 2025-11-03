// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title StudentRegistry
 * @dev Reliable student registry demonstrating good OOP-like structuring in Solidity.
 *
 * Features:
 *  - Register students with validation
 *  - Update marks (reads input from caller)
 *  - Efficient lookups (mapping + index)
 *  - Readable getters returning parallel arrays for UI-friendly output
 *  - Events for all important state changes
 */
contract StudentRegistry {

    // --------------------
    // STRUCT & STORAGE
    // --------------------
    struct Student {
        uint256 id;
        string name;
        uint8 age;        // age in years (0-255) — we validate realistic range
        string course;
        uint8 marks;      // marks from 0 to 100
    }

    Student[] private studentList;

    // Existence map and index map for O(1) access to a student's array index
    mapping(uint256 => bool) private registeredIds;
    mapping(uint256 => uint256) private studentIndex;

    // --------------------
    // EVENTS
    // --------------------
    event StudentRegistered(uint256 indexed id, string name, string course, uint8 age, uint8 marks);
    event MarksUpdated(uint256 indexed id, uint8 oldMarks, uint8 newMarks);
    event StudentRemoved(uint256 indexed id);

    // --------------------
    // CONSTANTS
    // --------------------
    uint8 private constant MIN_AGE = 3;     // minimum allowed age
    uint8 private constant MAX_AGE = 120;   // maximum allowed age
    uint8 private constant MAX_MARKS = 100; // marks cap

    // --------------------
    // MUTATING FUNCTIONS
    // --------------------

    /**
     * @notice Register a new student.
     * @param _id Unique student id (must be > 0).
     * @param _name Student name.
     * @param _age Student age (validated against MIN_AGE and MAX_AGE).
     * @param _course Course name.
     * @param _marks Initial marks (0..100). If unknown provide 0.
     */
    function registerStudent(
        uint256 _id,
        string memory _name,
        uint8 _age,
        string memory _course,
        uint8 _marks
    ) public {
        require(_id > 0, "Invalid ID");
        require(!registeredIds[_id], "Student ID already exists");
        require(_age >= MIN_AGE && _age <= MAX_AGE, "Age out of valid range");
        require(_marks <= MAX_MARKS, "Marks must be 0..100");

        studentList.push(Student({
            id: _id,
            name: _name,
            age: _age,
            course: _course,
            marks: _marks
        }));

        registeredIds[_id] = true;
        studentIndex[_id] = studentList.length - 1;

        emit StudentRegistered(_id, _name, _course, _age, _marks);
    }

    /**
     * @notice Update marks for an existing student.
     * @param _id Student ID to update.
     * @param _newMarks New marks value (0..100).
     */
    function updateMarks(uint256 _id, uint8 _newMarks) public {
        require(registeredIds[_id], "Student not registered");
        require(_newMarks <= MAX_MARKS, "Marks must be 0..100");

        uint256 idx = studentIndex[_id];
        uint8 old = studentList[idx].marks;
        studentList[idx].marks = _newMarks;

        emit MarksUpdated(_id, old, _newMarks);
    }

    /**
     * @notice Remove a student record.
     * @dev Uses swap-and-pop for efficient removal while keeping array compact.
     * @param _id Student ID to remove.
     */
    function removeStudent(uint256 _id) public {
        require(registeredIds[_id], "Student not registered");

        uint256 idx = studentIndex[_id];
        uint256 last = studentList.length - 1;

        if (idx != last) {
            Student memory lastStudent = studentList[last];
            studentList[idx] = lastStudent;
            studentIndex[lastStudent.id] = idx;
        }

        studentList.pop();
        delete registeredIds[_id];
        delete studentIndex[_id];

        emit StudentRemoved(_id);
    }

    // --------------------
    // VIEW / READ FUNCTIONS
    // --------------------

    //  * @notice Get a single student's details in a readable form.
    //  * @param _id Student ID to fetch.
    //  * @return id, name, age, course, marks
    
    function getStudentReadable(uint256 _id)
        public
        view
        returns (
            uint256 id,
            string memory name,
            uint8 age,
            string memory course,
            uint8 marks
        )
    {
        require(registeredIds[_id], "Student not registered");
        Student memory s = studentList[studentIndex[_id]];
        return (s.id, s.name, s.age, s.course, s.marks);
    }


    // * @notice Get all students as parallel arrays for friendly UI rendering.
    //* @return ids, names, ages, courses, marks
     
    function getAllStudentsReadable()
        public
        view
        returns (
            uint256[] memory ids,
            string[] memory names,
            uint8[] memory ages,
            string[] memory courses,
            uint8[] memory marks
        )
    {
        uint256 count = studentList.length;
        ids = new uint256[](count);
        names = new string[](count);
        ages = new uint8[](count);
        courses = new string[](count);
        marks = new uint8[](count);

        for (uint256 i = 0; i < count; i++) {
            Student memory s = studentList[i];
            ids[i] = s.id;
            names[i] = s.name;
            ages[i] = s.age;
            courses[i] = s.course;
            marks[i] = s.marks;
        }
    }

    /**
     * @notice Return number of registered students.
     * @return count number of students.
     */
    function getStudentCount() public view returns (uint256 count) {
        return studentList.length;
    }

    // --------------------
    // FALLBACKS
    // --------------------
    fallback() external payable {}
    receive() external payable {}
}
