resource "aws_iam_policy" "ecr_access_server" {
  name        = "${var.project_name}-ecr-access-server-policy"
  description = "Allows ECS tasks to pull images from the sie-server ECR repository."

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "ecr:GetAuthorizationToken",
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
        ],
        Resource = aws_ecr_repository.server.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_access_attachment_server" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = aws_iam_policy.ecr_access_server.arn
}

resource "aws_iam_policy" "ecr_access_router" {
  name        = "${var.project_name}-ecr-access-router-policy"
  description = "Allows ECS tasks to pull images from the sie-router ECR repository."

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetAuthorizationToken"
        ],
        Resource = aws_ecr_repository.router.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_access_attachment_router" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = aws_iam_policy.ecr_access_router.arn
}
