-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
-- Requirements:
-- Procedure ComputeAverageScoreForUser is taking 1 input:
-- user_id, a users.id value (you can assume user_id is linked to an existing users)
DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id INT
)
BEGIN
	UPDATE users 
END; //
DELIMITER ;
