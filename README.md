# cidpro_exam_generator
## What is this
Generate a CIDPRO examination question based on IDPRO's body of knowledge.

https://idpro.org/body-of-knowledge/

## generation example
```
% python3.9 main.py --url https://bok.idpro.org/article/id/64/
Using embedded DuckDB without persistence: data will be transient

Question: (Situation example: An organization wants to improve its account recovery (AR) process by implementing bearer tokens. What should they consider in terms of threats and mitigations?)
Choose the correct answer from the options below:
1. Bearer tokens can be shared freely among users for a simplified experience.
2. Bearer tokens' validity window should be as wide as possible for user convenience.
3. Minimize the validity window of bearer tokens and ensure the user remains on the same device and browser for better security.
4. Don't worry about the medium through which bearer tokens are sent, as they are inherently secure.

https://bok.idpro.org/article/id/64/


The correct answer is 3. Minimize the validity window of bearer tokens and ensure the user remains on the same device and browser for better security.

This is the correct answer because minimizing the validity window of bearer tokens reduces the risk of unauthorized access and abuse, such as through phishing. Ensuring the user remains on the same device and browser helps to verify that the user has not been phished for their information. This approach provides better security for the account recovery process while still allowing for a user-friendly experience.

https://bok.idpro.org/article/id/64/
```

## env
```
export OPENAI_API_KEY=""
export GPT_MODEL="gpt-4"
```
