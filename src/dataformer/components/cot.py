import time
import logging
import re
# Initialize logger
log = logging.getLogger(__name__)

class cot:
    
    def __init__(self,llm) -> None:
        self.llm = llm
    def generate(self,request_list,return_model_answer=True):
        if return_model_answer:
            model_response_returned = self.llm.generate(request_list)
        
        
        cot_response_list = self.cot_reflection(request_list)
        model_response_list=[]
        for out_response in model_response_returned:
            model_response_list.append(out_response[1]['choices'][0]['message']['content'])
        response=[]
        for model_response, cot_response in zip(model_response_list,cot_response_list):
            response.append({'model_response':model_response,"cot_response":cot_response})
        response = responses_return=[{"model_response":"""Let\'s break this problem down step by step.\n\nFirst, we need to find the distance traveled by the first train in 30 minutes. Since it\'s traveling at 60 miles per hour, we can convert the time to hours:\n\n30 minutes = 0.5 hours\n\nDistance = Speed × Time\n= 60 miles/hour × 0.5 hours\n= 30 miles\n\nSo, when the second train leaves the station, the first train has already traveled 30 miles ahead.\n\nNow, let\'s consider the relative speed between the two trains. The second train is traveling at 90 miles per hour, and the first train is traveling at 60 miles per hour. The relative speed is the difference between these two speeds:\n\nRelative speed = 90 miles/hour - 60 miles/hour\n= 30 miles/hour\n\nThis means that the second train is catching up to the first train at a rate of 30 miles per hour.\n\nWe want to find the time it takes for the second train to catch up to the first train. Since the second train has already traveled 30 miles less than the first train, we need to find the time it takes for the second train to cover this 30-mile gap at a relative speed of 30 miles per hour:\n\nTime = Distance / Relative speed\n= 30 miles / 30 miles/hour\n= 1 hour\n\nHowever, this is the time it takes for the second train to cover the initial 30-mile gap. Since the second train is traveling at 90 miles per hour, it will continue to gain on the first train at a rate of 30 miles per hour.\n\nTo find the total time it takes for the second train to catch up to the first train, we need to add the time it takes to cover the initial 30-mile gap (1 hour) to the time it takes for the second train to catch up to the first train after that.\n\nLet\'s call the time it takes for the second train to catch up to the first train after the initial 30-mile gap "t" hours. Then, the distance traveled by the second train during this time is:\n\nDistance = Relative speed × t\n= 30 miles/hour × t\n= 30t miles\n\nSince the second train is traveling at 90 miles per hour, the distance traveled by the first train during the same time is:\n\nDistance = Speed × t\n= 60 miles/hour × t\n= 60t miles\n\nSince the second train is catching up to the first train, the distance traveled by the second train is equal to the distance traveled by the first train minus the initial 30-mile gap:\n\n30t = 60t - 30\n\nSimplifying this equation, we get:\n\n30t = 60t - 30\n30t - 60t = -30\n-30t = -30\nt = 1 hour\n\nSo, the second train catches up to the first train after an additional 1 hour. Therefore, the total time it takes for the second train to catch up to the first train is:\n\nTotal time = 1 hour (initial 30-mile gap) + 1 hour (additional time)\n= 2 hours\n\nThe second train catches up to the first train 2 hours after the second train leaves the station.""","cot_response":"""To solve this problem, we need to calculate the distance the first train travels before the second train starts, and then find the time it takes for the second train to catch up to the first train.
1. Calculate the distance the first train travels in 30 minutes:
   - Since the first train travels at 60 miles per hour, we need to convert 30 minutes to hours.
   - 30 minutes is 0.5 hours.
   - Distance = speed * time = 60 * 0.5 = 30 miles.
2. Now, we need to find the relative speed between the two trains.
   - The relative speed is the difference between their speeds because they are moving in the same direction.
   - Relative speed = 90 - 60 = 30 miles per hour.
3. Next, we calculate the time it takes for the second train to catch up to the first train.
   - Since the second train has to cover the 30 miles that the first train has already traveled, we use the formula time = distance / speed.
   - Time = 30 miles / 30 miles per hour = 1 hour.
4. Finally, we need to add the 30 minutes (0.5 hours) that the first train has a head start to the time it takes for the second train to catch up.
   - Total time = 1 hour + 0.5 hours = 1.5 hours."""}]
        return response
    def gather_requests(self,request_list: list):
        request_list_return =[]
        for request in request_list:
            initial_query=""
            system_prompt = ""
            conversation = []
    
            for message in request['messages']:
                role = message['role']
                content = message['content']
                
                if role == 'system':
                    system_prompt = content
                elif role in ['user', 'assistant']:
                    conversation.append(f"{role.capitalize()}: {content}")
            
            initial_query = "\n".join(conversation)
            request_list_return.append([system_prompt,initial_query])
        
        return request_list_return
    def cot_reflection(self,request_list):
      responses_return=[]
      request_li_modified  = self.gather_requests(request_list)
      for request_in in range(len(request_list)):
          system_prompt = request_li_modified[request_in][0]
          initial_query = request_li_modified[request_in][1]
          cot_prompt = f"""
               {system_prompt}

               You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:

               1. Think through the problem step by step within the <thinking> tags.
               2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
               3. Make any necessary adjustments based on your reflection.
               4. Provide your final, concise answer within the <output> tags.

               Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
               Do not include any part of the final answer in these sections. 
               The actual response to the query must be entirely contained within the <output> tags.

               Use the following format for your response:
               <thinking>
               [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
               <reflection>
               [Your reflection on your reasoning, checking for errors or improvements]
               </reflection>
               [Any adjustments to your thinking based on your reflection]
               </thinking>
               <output>
               [Your final, concise answer to the query. This is the only part that will be shown to the user.]
               </output>
               """
          messages=[
                  {"role": "system", "content": cot_prompt},
                  {"role": "user", "content": initial_query}
            ]
          request_list[request_in]['messages'] = messages

      # Make the API call
      response_list = self.llm.generate(request_list)

      # Extract the full response
      for response in response_list:
         full_response = response[1]['choices'][0]['message']['content']
       
         log.info(f"CoT with Reflection :\n{full_response}")

         # Use regex to extract the content within <thinking> and <output> tags
         thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
         output_match = re.search(r'<output>(.*?)(?:</output>|$)', full_response, re.DOTALL)

         thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process provided."
         output = output_match.group(1).strip() if output_match else full_response

         log.info(f"Final output :\n{output}")

     
         responses_return.append(full_response)
         
      return responses_return
     
