// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ProductInventory
 * @dev A small inventory system that receives, sells, and displays products.
 */
contract ProductInventory {

    struct Product {
        uint256 id;
        string name;
        uint256 price;
        uint256 stock;
    }

    mapping(uint256 => Product) private productList;
    uint256[] private productIds;

    event ProductReceived(uint256 id, string name, uint256 quantity);
    event ProductSold(uint256 id, uint256 quantity, uint256 remainingStock);

    /// @notice Add new or restock an existing product
    function receiveProduct(uint256 _id, string memory _name, uint256 _price, uint256 _quantity) public {
        Product storage p = productList[_id];

        if (p.id == 0) {
            // new product
            productList[_id] = Product(_id, _name, _price, _quantity);
            productIds.push(_id);
        } else {
            // restock existing product
            p.stock += _quantity;
        }
        emit ProductReceived(_id, _name, _quantity);
    }

    /// @notice Sell a product
    function sellProduct(uint256 _id, uint256 _quantity) public {
        Product storage p = productList[_id];
        require(p.id != 0, "Product not found!");
        require(p.stock >= _quantity, "Insufficient stock!");
        p.stock -= _quantity;
        emit ProductSold(_id, _quantity, p.stock);
    }

    /// @notice Display details of a single product
    function getProduct(uint256 _id) public view returns (Product memory) {
        require(productList[_id].id != 0, "Product not found!");
        return productList[_id];
    }

    /// @notice Display all products
    function getAllProducts() public view returns (Product[] memory) {
        Product[] memory all = new Product[](productIds.length);
        for (uint256 i = 0; i < productIds.length; i++) {
            all[i] = productList[productIds[i]];
        }
        return all;
    }
}
