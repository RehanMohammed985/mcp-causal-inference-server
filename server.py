# server.py
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
import dowhy
import re
import warnings
import json
# Create an MCP server
mcp = FastMCP("Demo")

# Generate synthetic customer spending data
num_users = 10000
num_months = 12

signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0, 2, size=num_users)
df = pd.DataFrame({
    'user_id': np.repeat(np.arange(num_users), num_months),
    'signup_month': np.repeat(signup_months, num_months),
    'month': np.tile(np.arange(1, num_months + 1), num_users),
    'spend': np.random.poisson(500, num_users * num_months)
})

df["treatment"] = df["signup_month"] > 0

df["spend"] = df["spend"] - df["month"] * 10

# Apply treatment effect
after_signup = (df["signup_month"] < df["month"]) & (df["treatment"])
df.loc[after_signup, "spend"] = df[after_signup]["spend"] + 100

# Define causal graph
causal_graph = """digraph {
    treatment[label="Program Signup in month i"];
    pre_spends;
    post_spends;
    Z->treatment;
    pre_spends -> treatment;
    treatment->post_spends;
    signup_month->post_spends;
    signup_month->treatment;
}"""

# Process data for a given signup month (e.g., i=3)
i = 3
df_i_signupmonth = (
    df[df.signup_month.isin([0, i])]
    .groupby(["user_id", "signup_month", "treatment"])
    .apply(
        lambda x: pd.Series({
            "pre_spends": x.loc[x.month < i, "spend"].mean(),
            "post_spends": x.loc[x.month > i, "spend"].mean(),
        })
    )
    .reset_index()
)



@mcp.tool()

def get_causal_estimate(treatment: str, outcome: str) -> str:
    """Calculate the estimated treatment effect using an appropriate backdoor method."""

    if not treatment or not outcome:
        return f"Error: Treatment ({treatment}) or Outcome ({outcome}) is not defined correctly."

    # Build the causal model
    model = dowhy.CausalModel(
        data=df_i_signupmonth,  # Use pre-processed subset of data
        graph=causal_graph.replace("\n", " "),
        treatment=treatment,
        outcome=outcome
    )

    try:
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    except Exception as e:
        return f"Error identifying causal effect: {e}"

    # Try supported estimation methods in order
    estimation_methods = [
        "backdoor.propensity_score_matching",
        "backdoor.linear_regression"
    ]

    for method in estimation_methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress FutureWarnings
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method,
                    target_units="att"
                )
            return f"Estimated causal effect using {method}: {estimate.value}"
        except Exception:
            continue

    return f"No suitable estimation method found for {treatment} → {outcome}."



# Fixing the query_relationship function
@mcp.tool()
def query_relationship(
    treatment: str,
    outcome: str
) -> str:
    """Automatically determines whether the causal effect betweeb treatment and outcome variables can be calculated and returns the method that can be used to estimate this effect."""

    # Retrieve all variables and their descriptions from the dataset
    var_desc = get_variable_descriptions()

    err_msg = ""
    if treatment not in var_desc.keys():
        err_msg += "Name of treatment variable was not detected correctly.\n"

    if outcome not in var_desc.keys():
        err_msg += "Name of outcome variable was not detected correctly."

    if len(err_msg) > 0:
        err_msg += "Here is the list of the variables."
        err_msg += json.dumps(var_desc)
        return err_msg


    try:
        # Build a causal model using DoWhy
        model = dowhy.CausalModel(
            data=df,  # Use the dataset
            graph=causal_graph.replace("\n", " "),  # Set causal relationships
            treatment=treatment,  # Define treatment variable
            outcome=outcome  # Define outcome variable
        )

        # Identify the causal effect, even if the model is unidentifiable
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # print(list(identified_estimand.estimands.keys()))

        ret_msg = ""
        if identified_estimand.estimands["backdoor"] is not None:
            ret_msg += "backdoor, "
        if identified_estimand.estimands["frontdoor"]:
            ret_msg += "frontdoor, "
        if identified_estimand.estimands["iv"]:
            ret_msg += "instrumental variable"
        if len(ret_msg) > 0:
            ret_msg = "The causal estimate is identifieable and it can be obtained using these criteria: " + ret_msg + "."
            return ret_msg
        else:
            return f"No identifiable causal effect for {treatment} → {outcome} using standard methods."

    except Exception as e:
        # Catch any errors and return an informative message
        return f"Error constructing causal model. No identifiable causal effect for {treatment} → {outcome} using standard methods."



@mcp.tool()
def get_variable_descriptions() -> dict:
    """Automatically detects variable names and their descriptions in the dataset."""

    descriptions = {
        "treatment": "Treatment indicating whether the user signed up for the program",
        "pre_spends": "Amount spent before the treatment",
        "post_spends": "Amount spent after the treatment",
        "Z": "This is just a confound variable",
        "signup_month": "The month when the user signed up for the program"
    }

    return descriptions


# Start the MCP Server
if __name__ == "__main__":
    mcp.run()
