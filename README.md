## Simple linear regression
data(iris) 
# Fit the simple linear regression model 
model <- lm(Sepal.Length ~ Sepal.Width, data = iris) 
# Print the model summary 
summary(model) 

## Multiple linear regression
# Fit the multiple linear regression model 
model <- lm(mpg ~ wt + disp + hp, data = mtcars) 
# Print the model summary 
summary(model)

## Logistic regression
# Fit the logistic regression model 
model <- glm(vs ~ mpg + wt + hp, data = mtcars, family = binomial) 
# Print the model summary 
summary(model)

## Plotting the regression line 
ggplot(mtcars, aes(x=wt, y=mpg)) + 
geom_point() + 
geom_smooth(method=lm, se=FALSE, color="red", linetype="dashed") + 
theme_minimal() + 
labs(title="Simple Linear Regression: MPG vs Weight", 
x="Weight", 
y="Miles per Gallon", 
caption="Red line represents the regression line")

## Printing out the residual & fitted values
multiple_model <- lm(mpg ~ wt + hp, data = mtcars) 
summary(multiple_model) 
# Print the fitted values 
fitted_values <- multiple_model$fitted.values 
print(fitted_values)
# Print the residuals 
residual_values <- multiple_model$residuals 
print(residual_values) 
# Create a data frame containing observed mpg, fitted values, and residuals 
results <- data.frame(  Observed = mtcars$mpg,Fitted = fitted_values, Residuals = residual_values) 
# View the first few rows of the results 
head(results) 

## augment(multiple_model) 
ggplot(augment(multiple_model), aes(.fitted, .resid)) + 
geom_point() + 
geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
theme_minimal() + 
labs(title="Residuals vs Fitted Values", 
x="Fitted Values", 
y="Residuals", 
caption="Red line represents y=0") 

## SUPPORT VECTOR REGRESSION (SVR)
library(e1071) 
library(ggplot2) 
svr_model <- svm(mpg ~ wt, data = mtcars, kernel = "radial") 
summary(svr_model)
# Load the mtcars dataset 
data("mtcars") 
# Fit the SVR model predicting mpg based on wt and hp 
svr_model <- svm(mpg ~ wt + hp, data = mtcars, type = "eps-regression", 
kernel = "radial") 

## CONTOUR PLOT
wt_seq <- seq(min(mtcars$wt), max(mtcars$wt), length.out = 100) 
hp_seq <- seq(min(mtcars$hp), max(mtcars$hp), length.out = 100) 
grid <- expand.grid(wt = wt_seq, hp = hp_seq) 
grid$mpg <- predict(svr_model, newdata = grid) 
# Basic plot of the fitted surface 
ggplot(grid, aes(x = wt, y = hp, fill = mpg)) + 
  geom_tile() +  
  geom_contour(aes(z = mpg), color = "white") + 
  labs(title = "SVR Model Prediction of mpg", 
       x = "Weight (1000 lbs)", 
       y = "Horsepower", 
       fill = "Miles per Gallon") + 
  theme_minimal()+ 
# Optionally add the actual data points 
geom_point(data = mtcars, aes(x = wt, y = hp, color = mpg), size = 3)

## INDICES OF THE SUPPORT VECTOR
# Identify the indices of the support vectors 
support_vector_indices <- svr_model$index 
# Extract the support vectors from the original data 
support_vectors <- mtcars[support_vector_indices, ] 
# Add the support vectors to the plot with a different shape or color to 
distinguish them 
ggplot(grid, aes(x=wt,y=hp,fill=mpg))+ 
  geom_tile()+ 
  geom_contour(aes(z=mpg),color = "white")+ 
  geom_point(data = mtcars, aes(x=wt,y=hp),color= "white", size=5)+ 
  geom_point(data = support_vectors,aes(x=wt,y=hp), color = "red", size = 5, 
shape=8)+ 
    # Red squares for support vectors 
  labs(title = "SVR Model Prediction of mpg", 
       x = "Weight (1000 lbs)", 
       y = "Horsepower", 
       fill = "Miles per Gallon") + 
  theme_minimal()
predictions <- predict(svr_model, mtcars) 
mse <- mean((mtcars$mpg - predictions)^2) 
print(paste("MSE: ", mse))

## ARTIFITIAL NEURAL NETWORK

library(neuralnet) 
nn_model <- neuralnet(mpg ~ wt + hp, data = mtcars, hidden = 2) 
print(nn_model) 
# Plotting the Neural Network 
# Let’s plot the neural network structure. 
plot(nn_model) 
# Making Predictions 
# We can use the model to make predictions on the dataset. 
predictions <- compute(nn_model, mtcars[,c("wt", "hp")]) 
head(predictions$net.result) 












