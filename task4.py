
from dataclasses import dataclass
from typing import Optional
import random
import sys

from task1 import faqs
from task2 import LLMClient, API_KEY, MODEL
from task3 import hybrid_search


# Conversation Turn
@dataclass
class ConversationTurn:
    role: str
    content: str
    faq_id: Optional[str] = None
    confidence: Optional[float] = None


# Support Agent Class
class SupportAgent:

    def __init__(self, faqs, llm_client, confidence_threshold=0.15):

        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        self.conversation_history = []
        self.escalated = False
        self.low_confidence_streak = 0

    def handle_message(self, user_message):

        # Record user message
        self.conversation_history.append(
            ConversationTurn("user", user_message)
        )

        # Search using Task 3
        results = hybrid_search(self.faqs, user_message, 1)

        if results and results[0][1] >= self.confidence_threshold:

            faq, confidence = results[0]

            # Generate response using Task 2
            response = self.llm_client.generate_faq_response(
                user_message, faq
            )

            faq_id = faq["id"]

            # Reset low confidence counter
            self.low_confidence_streak = 0

        else:

            # Fallback response
            response = (
                "I don't have information about this question "
                "in my knowledge base.\n"
                "Would you like me to connect you with a "
                "human support agent?"
            )

            faq_id = None
            confidence = 0.0

            self.low_confidence_streak += 1

        # Record assistant response
        self.conversation_history.append(
            ConversationTurn(
                "assistant",
                response,
                faq_id,
                confidence
            )
        )

        return response

    def escalate(self, reason="User requested human support"):

        self.escalated = True

        ticket_id = f"TICKET-{random.randint(10000, 99999)}"

        return (
            "Your request has been escalated to our support team.\n"
            f"Reason: {reason}\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours."
        )

    def get_conversation_summary(self):

        if not self.conversation_history:
            return "No conversation history available."

        summary = ["\n========== Conversation History =========="]

        for turn in self.conversation_history:

            summary.append(
                f"\n{turn.role.upper()}: {turn.content}"
            )

            if turn.role == "assistant":

                summary.append(
                    f"FAQ ID: {turn.faq_id or 'None'}"
                )

                summary.append(
                    f"Confidence: {turn.confidence}"
                )

        summary.append(
            f"\nEscalated: {'Yes' if self.escalated else 'No'}"
        )

        summary.append("==========================================")

        return "\n".join(summary)

    def reset(self):

        self.conversation_history.clear()
        self.escalated = False
        self.low_confidence_streak = 0


# Create SupportAgent
def create_agent():

    client = LLMClient(API_KEY, MODEL)

    return SupportAgent(faqs, client)


# Interactive Chat Interface
def chat_interface():

    agent = create_agent()

    print("\n╔══════════════════════════════════════════╗")
    print("║       SupportAI — Helpdesk Agent         ║")
    print("╚══════════════════════════════════════════╝")

    print("\nWelcome! How can I help you today?")
    print("Commands: history | escalate | reset | quit")

    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        # Quit
        if command == "quit":

            print("\nThank you for using SupportAI. Goodbye!")
            break

        # History
        elif command == "history":

            print(agent.get_conversation_summary())

        # Escalation
        elif command == "escalate":

            print("\nSupportAI:")
            print(agent.escalate())

        # Reset
        elif command == "reset":

            agent.reset()
            print("\nSupportAI: Conversation has been reset.")

        # Normal question
        else:

            response = agent.handle_message(user_input)

            last_turn = agent.conversation_history[-1]

            print(
                f"\nSupportAI [confidence: {last_turn.confidence}, "
                f"faq: {last_turn.faq_id or 'None'}]:"
            )

            print(response)

            # Suggest escalation after 3 low-confidence answers
            if agent.low_confidence_streak >= 3:

                print(
                    "\nSupportAI: I've been unable to find "
                    "the right answers. "
                    "Please type 'escalate' for human support."
                )


# End-to-End Demonstration
def run_demo():

    agent = create_agent()

    print("\n==========================================")
    print("       SUPPORTAI END-TO-END DEMO")
    print("==========================================")

    # Scenario 1: Clear FAQ question
    print("\nSCENARIO 1: Clear FAQ question")

    question = "How do I reset my password?"
    print("You:", question)

    response = agent.handle_message(question)
    last = agent.conversation_history[-1]

    print(f"SupportAI [confidence: {last.confidence}, faq: {last.faq_id}]:")
    print(response)

    # Scenario 2: Paraphrased question
    print("\nSCENARIO 2: Paraphrased question")

    question = "I forgot my login credentials"
    print("You:", question)

    response = agent.handle_message(question)
    last = agent.conversation_history[-1]

    print(f"SupportAI [confidence: {last.confidence}, faq: {last.faq_id}]:")
    print(response)

    # Scenario 3: Question outside FAQs
    print("\nSCENARIO 3: Question outside FAQs")

    question = "What are your office hours in Tokyo?"
    print("You:", question)

    response = agent.handle_message(question)
    last = agent.conversation_history[-1]

    print(f"SupportAI [confidence: {last.confidence}, faq: None]:")
    print(response)

    # Scenario 4: User requests escalation
    print("\nSCENARIO 4: User requests escalation")

    print("You: escalate")
    print("SupportAI:")
    print(agent.escalate())

    print("\n==========================================")
    print("       END OF DEMONSTRATION")
    print("==========================================")


# Main Program
if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        chat_interface()
