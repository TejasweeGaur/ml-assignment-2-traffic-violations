# Dataset Features Analysis

## Raw Feature Analysis

| Feature | Feature Type | Usability | Verdict |
| --- | --- | --- | --- |
| Violation_ID | Numerical | Unique identifier for each violation | Can be Dropped |
| Violation_Type | Categorical | Target variable indicating type of violation | Keep as it is |
| Fine_Amount | Numerical | Indicates severity but post-violation information | Can be Dropped |
| Location | Categorical | Geospatial analysis; encode into state codes | Keep but need Encoding |
| Date | Datetime | Extract day, month, year for analyzing weekends, holidays, etc. | Keep but need Feature Engineering |
| Time | Datetime | Extract hour, minute for analyzing early mornings, late nights, etc. | Keep but need Feature Engineering |
| Vehicle_Type | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| Vehicle_Color | Categorical | Classification analysis; encode into color categories | Keep but need Encoding |
| Vehicle_Model_Year | Numerical | Numerical Feature Capturing on vehicle age | Keep as it is |
| Registration_State | Categorical | Geospatial analysis; encode into state codes | Keep but need Encoding |
| Driver_Age | Numerical | Numerical Feature Capturing on driver demographics | Keep as it is |
| Driver_Gender | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| License_Type | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| Penalty_Points | Numerical | Numerical Feature Capturing on violation severity | Keep as it is |
| Weather_Condition | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| Road_Condition | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| Officer_ID | Categorical | Officer identifier; minimal predictive value for violation type prediction | Can be Dropped |
| Issuing_Agency | Categorical | Post-violation information; minimal predictive value | Can be Dropped |
| License_Validity | Categorical | Classification analysis; indicates license status | Keep but need Encoding |
| Number_of_Passengers | Numerical | Numerical Feature Capturing on vehicle occupancy | Keep as it is |
| Helmet_Worn | Categorical | Classification analysis; contains missing values | Keep but need Imputation |
| Seatbelt_Worn | Categorical | Classification analysis; contains missing values | Keep but need Imputation |
| Traffic_Light_Status | Categorical | Classification analysis; will be handled by one-hot encoding in preprocessing pipeline | Keep but need Encoding |
| Speed_Limit | Numerical | Numerical Feature Capturing on traffic regulations | Keep as it is |
| Recorded_Speed | Numerical | Numerical Feature Capturing on actual speed | Keep as it is |
| Alcohol_Level | Numerical | Numerical Feature Capturing on intoxication; contains missing values | Keep but need Imputation |
| Breathalyzer_Result | Categorical | Classification analysis; contains missing values | Keep but need Imputation |
| Towed | Categorical | Classification analysis; indicates violation severity | Keep but need Encoding |
| Fine_Paid | Categorical | Post-violation information; can be dropped | Can be Dropped |
| Payment_Method | Categorical | Post-violation information; can be dropped | Can be Dropped |
| Court_Appearance_Required | Categorical | Post-violation information; can be dropped | Can be Dropped |
| Previous_Violations | Numerical | Numerical Feature Capturing on violation history | Keep as it is |
| Comments | Text | Text feature with minimal predictive value | Can be Dropped |

---

## Features to Drop

| Feature | Reason for Dropping |
| --- | --- |
| Violation_ID | Unique identifier for each violation |
| Fine_Amount | Post-violation information; indicates severity but determined after violation occurs |
| Officer_ID | Officer identifier; minimal predictive value for violation type prediction |
| Issuing_Agency | Post-violation information; minimal predictive value |
| Fine_Paid | Post-violation information; determined after violation occurs |
| Payment_Method | Post-violation information; determined after violation occurs |
| Court_Appearance_Required | Post-violation information; determined after violation occurs |
| Comments | Text feature with minimal predictive value |
