from transformers import pipeline
import tensorflow as tf

qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad",
    tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad"
)

question, context = "When does the closing ceremony start?", """We wish you and your parents good health and good luck
in the new year. I would like to inform you that the closing ceremony and graduation ceremony for the 2018 school year
and the 2019 school year and the entrance ceremony for new students will be announced as follows. Please guide us so
that the time is safe and enjoyable without being exposed to various harmful environments. 1. Closing Ceremony and
Diploma Award Ceremony Schedule: January 4, 2019 (Friday), students return to school at 9:00 1) Closing Ceremony: 1st
and 2nd graders, each classroom, 9:00-10:50 (no lunch) 2) Diploma award ceremony: 3rd year, multi-purpose auditorium
(2nd floor),10:30-12:20 2. School starts and freshman entrance ceremony: March 4 (Mon), 2019, students return to school
at 9:00 1) March 4 Schedule: Normal classes on Mondays ( Lunch) (Prepared materials: textbooks, notebooks, writing
implements, slippers for students (white), etc.) 2) Entrance ceremony for new students: Multipurpose auditorium (2nd
floor) 10:30, New students arrive at 9:00 (entry into the new class classroom) 3) New class and Timetable announcement
- Admission students: 2019.2.28. To be posted on the school website - Enrolled students: 2019.2.28. To be posted on the
school website *The above schedule is subject to change due to school circumstances. Please refer to the school website.
"""

print(qa_pipeline({
    'context': context,
    'question': question

}))
