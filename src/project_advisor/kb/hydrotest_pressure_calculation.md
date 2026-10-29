# Hydrotest Pressure Calculation

Synthetic Meridian Project Delivery Advisor demo, snapshot 2026-10-29.

Demo rule modeled on a hydrostatic piping test: test pressure Pt = 1.5 x design pressure P x (St / S), where St is the allowable stress at test temperature and S is the allowable stress at design temperature. Cap the ratio St/S at 6.5. SP-1000-8820 has P = 40 barg, St = 138 MPa and S = 110.4 MPa, so St/S = 1.25 and Pt = 1.5 x 40 x 1.25 = 75 barg. Using 1.5 x P alone gives 60 barg, which is wrong here. This is an illustration, not an approved test procedure, test pack or permit.
