"""Team Laplace policy: online-learned reciprocal value model, ranked clarification, global matching.

Layers are kept separate (PS section 15): `observe` normalises the request, `model` estimates pair
values from observable feedback only, `asks` and `matching` decide, `validate` re-checks every action.
"""
