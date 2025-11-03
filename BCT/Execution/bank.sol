// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SmartBank
 * @dev Robust decentralized banking system demonstrating OOP principles.
 *      Features:
 *       - Account creation and deletion
 *       - Deposits and withdrawals
 *       - Account-to-account transfers
 *       - Full list of all account holders
 */
contract SmartBank {

    // -------------------- DATA STRUCTURES --------------------
    struct Account {
        string name;
        address owner;
        uint256 balance;
        bool isActive;
    }

    mapping(address => Account) private accounts;
    address[] private accountHolders;

    // -------------------- EVENTS --------------------
    event AccountCreated(address indexed owner, string name);
    event DepositMade(address indexed owner, uint256 amount, uint256 newBalance);
    event WithdrawalMade(address indexed owner, uint256 amount, uint256 newBalance);
    event TransferMade(address indexed from, address indexed to, uint256 amount);
    event AccountClosed(address indexed owner, string name);

    // -------------------- MODIFIERS --------------------
    modifier onlyAccountHolder() {
        require(accounts[msg.sender].isActive, "No active account found for caller!");
        _;
    }

    // -------------------- FUNCTIONS --------------------

    /// @notice Create a new account
    /// @param _name Name of the account holder
    function createAccount(string memory _name) external {
        require(!accounts[msg.sender].isActive, "Account already exists!");
        require(bytes(_name).length > 0, "Name cannot be empty");

        accounts[msg.sender] = Account({
            name: _name,
            owner: msg.sender,
            balance: 0,
            isActive: true
        });

        accountHolders.push(msg.sender);
        emit AccountCreated(msg.sender, _name);
    }

    /// @notice Deposit funds into your account
    function deposit() external payable onlyAccountHolder {
        require(msg.value > 0, "Deposit amount must be greater than zero");
        accounts[msg.sender].balance += msg.value;
        emit DepositMade(msg.sender, msg.value, accounts[msg.sender].balance);
    }

    /// @notice Withdraw funds from your account
    /// @param _amount The amount to withdraw in wei
    function withdraw(uint256 _amount) external onlyAccountHolder {
        Account storage acc = accounts[msg.sender];
        require(_amount > 0, "Invalid withdrawal amount");
        require(acc.balance >= _amount, "Insufficient balance");

        acc.balance -= _amount;
        payable(msg.sender).transfer(_amount);
        emit WithdrawalMade(msg.sender, _amount, acc.balance);
    }

    /// @notice Transfer funds to another account
    /// @param _to Recipient address
    /// @param _amount Amount to transfer in wei
    function transfer(address _to, uint256 _amount) external onlyAccountHolder {
        require(_to != msg.sender, "Cannot transfer to self");
        require(accounts[_to].isActive, "Recipient does not have an active account");

        Account storage sender = accounts[msg.sender];
        Account storage receiver = accounts[_to];

        require(_amount > 0, "Transfer amount must be greater than zero");
        require(sender.balance >= _amount, "Insufficient funds for transfer");

        sender.balance -= _amount;
        receiver.balance += _amount;

        emit TransferMade(msg.sender, _to, _amount);
    }

    /// @notice View your account details
    function viewAccount()
        external
        view
        onlyAccountHolder
        returns (string memory, address, uint256)
    {
        Account memory acc = accounts[msg.sender];
        return (acc.name, acc.owner, acc.balance);
    }

    /// @notice Close your account permanently
    function closeAccount() external onlyAccountHolder {
        Account storage acc = accounts[msg.sender];
        require(acc.balance == 0, "Withdraw funds before closing the account");

        acc.isActive = false;
        emit AccountClosed(msg.sender, acc.name);
    }

    /// @notice View all account holders
    function getAllAccounts() external view returns (Account[] memory) {
        uint256 activeCount = 0;

        // Count active accounts
        for (uint256 i = 0; i < accountHolders.length; i++) {
            if (accounts[accountHolders[i]].isActive) {
                activeCount++;
            }
        }

        // Create an array of active accounts
        Account[] memory list = new Account[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < accountHolders.length; i++) {
            if (accounts[accountHolders[i]].isActive) {
                list[index] = accounts[accountHolders[i]];
                index++;
            }
        }
        return list;
    }

    // -------------------- FALLBACKS --------------------
    fallback() external payable {}
    receive() external payable {}
}
