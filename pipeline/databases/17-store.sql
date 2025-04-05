-- creates a trigger that decreases the quantity of an item after adding a new order.
-- Quantity in the table items can be negative.
DROP TRIGGER IF EXISTS itm_qty_decrease;
DELIMITER $$
CREATE TRIGGER itm_qty_decrease
    AFTER INSERT ON orders
    FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE items.name=new.item_name;
END$$
DELIMITER ;